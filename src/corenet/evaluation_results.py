# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence

import PIL.Image
import io
import pandas
import torch as t
import torch.utils
import torch.utils.tensorboard

from corenet import configuration
from corenet import distributed as dist_util
from corenet import file_system as fs
from corenet import misc_util
from corenet import voxel_metrics
from corenet.data import batched_example
from corenet.data import dataset as dataset_lib
from corenet.visualization import artifacts as visualization_artifacts
from corenet.visualization import colors


def extract_labels(pdf: t.Tensor, b: batched_example.BatchedExample,
                   task_type: configuration.TaskType):
  """Extracts labels from predicted class probability and GT for a task type."""

  pred_labels = pdf.argmax(dim=1).to(t.int32)
  gt = b.grid
  if task_type == configuration.TaskType.FG_BG:
    mesh_labels = t.cat(b.mesh_labels)[:, None, None, None]
    pred_labels = pred_labels * mesh_labels
    gt = gt * mesh_labels

  return pred_labels, gt


def visualize_output(
    pdf: t.Tensor,
    ex: batched_example.BatchedExample,
    task_type: configuration.TaskType,
    batch_indices: Iterable[int] = None
) -> List[List[t.Tensor]]:
  """Visualizes the predicted outputs next to the ground-truth.

  Args:
    pdf: The class probabilities output by the model.
    ex: The input batch, containing the ground-truth among others.
    task_type: The task type.
    batch_indices: The batch elements to visualized. Visualizes all by default.

  Returns:
    One list of images for each batch element, containing the element
    visualized from different angles.
  """

  palette = misc_util.to_tensor(colors.DEFAULT_COLOR_PALETTE, t.float32)
  palette = palette.to(pdf.device)
  output_images = []
  scene_num_tri = [n.sum().item() for n in ex.mesh_num_tri]
  offsets = t.tensor(scene_num_tri).cumsum(0).tolist()
  offsets = [0] + offsets[:-1]

  if not batch_indices:
    batch_indices = range(pdf.shape[0])

  vis = visualization_artifacts
  pred_lbl, gt_lbl = extract_labels(pdf, ex, task_type)
  for batch_idx in batch_indices:
    v2x = ex.v2x_transform[batch_idx].inverse()
    gt_mesh_labels = ex.mesh_labels[batch_idx]
    artifacts_3d = []

    # Marching cubes visualization for the predicted volume
    if task_type == configuration.TaskType.FG_BG:
      assert gt_mesh_labels.shape == (1,)
      mc_colors = palette.new_tensor([0, gt_mesh_labels[0]], dtype=t.int64)
      mc_colors = palette[mc_colors]
    else:
      num_classes = pdf.shape[1]
      mc_colors = palette[:num_classes]
    artifacts_3d.append(vis.MarchingCubesArtifact(
        pdf[batch_idx], v2x, mc_colors))

    # Mesh visualization of the GT scene
    gt_mesh_colors = palette[gt_mesh_labels.to(t.int64)]
    mesh_num_tri = ex.mesh_num_tri[batch_idx]
    offset = offsets[batch_idx]
    gt_vertices = ex.vertices[offset:offset + scene_num_tri[batch_idx]]
    artifacts_3d.append(vis.MultiMeshArtifact(gt_vertices, mesh_num_tri,
                                              mesh_colors=gt_mesh_colors))

    # Voxel grid visualizations of both the predicted volume and the GT
    artifacts_3d.append(vis.VoxelGridArtifact(pred_lbl[batch_idx], v2x))
    artifacts_3d.append(vis.VoxelGridArtifact(gt_lbl[batch_idx], v2x))

    artifacts = [vis.ImageArtifact(ex.input_image[batch_idx]), artifacts_3d]
    camera_images = vis.visualize_artifacts(
        artifacts, ex.camera_transform[batch_idx], ex.view_transform[batch_idx])
    output_images.append(camera_images)
  return output_images


class QualitativeResults:
  """Distributed helper for rendering and managing of qualitative results."""

  def __init__(self, eval_config: configuration.EvalConfig,
               dataset: dataset_lib.CoReNetDataset, image_output_dir: str):
    self.eval_config = eval_config
    # Always show results for the same scenes in the dataset. To show different
    # scenes for different eval runs, you need to shuffle the input dataset.

    self.disk_result_ids = {
        v.scene_id for v in dataset[:eval_config.num_qualitative_results]}
    self.tb_result_ids = {
        v.scene_id
        for v in dataset[:eval_config.num_qualitative_results_in_tensor_board]}
    self.ids_of_interest = self.tb_result_ids | self.disk_result_ids

    self.tb_results = {}  # type: Dict[str, List[t.Tensor]]
    self.image_output_dir = image_output_dir

  def _write_image(self, scene_id: str, scene_images: List[t.Tensor]):
    scene_id = scene_id.replace("/", "_")
    image = t.cat(scene_images, dim=0)
    pil_image = PIL.Image.fromarray(image.cpu().numpy())
    pil_image.save(fl := io.BytesIO(), format="png")
    fn = fs.join(self.image_output_dir, f"img_{scene_id}.png")
    fs.make_dirs(fs.dirname(fn))
    fs.write_bytes(fn, fl.getvalue())

  def add_batch(self, pdf: t.Tensor, ex: batched_example.BatchedExample):
    """Adds qualitative results from the current batch."""

    batch_indices = [i for i, v in enumerate(ex.scene_id)
                     if v in self.ids_of_interest]
    if not batch_indices:
      return
    task_type = self.eval_config.data.voxelization_config.task_type
    all_images = visualize_output(pdf, ex, task_type, batch_indices)
    scene_ids = [ex.scene_id[i] for i in batch_indices]

    # Write the results on disk. Store the relevant ones for TensorBoard
    for scene_id, scene_images in zip(scene_ids, all_images):
      if scene_id in self.disk_result_ids:
        self._write_image(scene_id, scene_images)
      if scene_id in self.tb_result_ids:
        self.tb_results[scene_id] = scene_images

  def write_tensor_board_summary(
      self, sw: Optional[torch.utils.tensorboard.SummaryWriter],
      global_step: int
  ):
    """Writes the saved images to a tensorboard summary."""
    all_results = dist_util.gather(self.tb_results, 0)
    if dist_util.info().global_rank == 0 and sw:
      all_results = {k: v for d in all_results for k, v in d.items()}
      all_results = sorted(all_results.items(), key=lambda v: v[0])
      for rec_idx, (scene_id, scene_images) in enumerate(all_results):
        for cam_idx, image in enumerate(scene_images):
          assert (len(image.shape) == 3 and image.shape[-1] == 3 and
                  image.dtype == t.uint8)
          image = image.permute([2, 0, 1])
          sw.add_image(f"rec_{rec_idx}/cam_{cam_idx}", image, global_step)
    else:
      assert sw is None


GLOBAL_CLASS_NAME = "__global__"


def compute_voxel_metrics(
    confusion_matrix: t.Tensor, classes: List[str]
) -> pandas.DataFrame:
  """Computes voxel metrics from a confusion matrix.

  Args:
    confusion_matrix: The confusion matrix, with dimensions (gt, predicted).
    classes: The class names

  Returns:
    A pandas data frame with the metrics. Columns are the classes and an
    additional "__global__" column, containing the class-agnostic FG/BG metrics.
    Rows are the metrics (iou, precision, recall)

  """
  tfpn = voxel_metrics.compute_tfpn(confusion_matrix)
  tfpn_fg = voxel_metrics.compute_tfpn_fg(confusion_matrix)
  metrics = voxel_metrics.compute_voxel_metrics(tfpn).cpu().numpy()
  fg_metrics = voxel_metrics.compute_voxel_metrics(tfpn_fg).cpu().numpy()
  metrics = pandas.DataFrame(dataclasses.asdict(metrics), index=classes).T
  fg_metrics = pandas.DataFrame(
      dataclasses.asdict(fg_metrics), index=[GLOBAL_CLASS_NAME]).T
  return pandas.concat([metrics, fg_metrics], axis=1)


def log_voxel_metrics_to_tensorboad(
    writer: torch.utils.tensorboard.SummaryWriter,
    metrics: pandas.DataFrame,
    global_step: int
):
  """Logs metrics in a data frame to a tensorboard summary writer."""
  assert metrics.columns[-1] == GLOBAL_CLASS_NAME
  assert metrics.columns[0] == dataset_lib.VOID_LABEL_NAME

  class_metrics = metrics.iloc[:, :-1].T
  for row in class_metrics.iterrows():
    k, v = row
    writer.add_scalar(f"IoU/{k}", v.iou, global_step)
    writer.add_scalar(f"Precision/{k}", v.precision, global_step)
    writer.add_scalar(f"Recall/{k}", v.recall, global_step)

  means = metrics.iloc[:, 1:-1].T.mean()
  writer.add_scalar("General/mIoU", means.iou, global_step)
  writer.add_scalar("General/mPrecision", means.precision, global_step)
  writer.add_scalar("General/mRecall", means.recall, global_step)

  fgbg_metrics = metrics.iloc[:, -1]
  writer.add_scalar("General/fgbgIoU", fgbg_metrics.iou, global_step)
  writer.add_scalar("General/fgbgPrecision", fgbg_metrics.precision,
                    global_step)
  writer.add_scalar("General/fgbgRecall", fgbg_metrics.recall, global_step)


class QuantitativeResults:
  def __init__(self, classes: Sequence[str],
               eval_config: configuration.EvalConfig):
    self.classes = list(classes)
    self.confusion_matrix = t.zeros([len(classes)] * 2, device="cuda")
    self.config = eval_config
    self.voxel_metrics_df = None

  def add_batch(self, pdf: t.Tensor, batch: batched_example.BatchedExample):
    pred_labels, gt_labels = extract_labels(
        pdf, batch, self.config.data.voxelization_config.task_type)
    self.confusion_matrix += voxel_metrics.confusion_matrix(
        pred_labels, gt_labels, len(self.classes))

  def compute_metrics(self):
    t.distributed.reduce(
        self.confusion_matrix, 0, op=t.distributed.ReduceOp.SUM)
    if dist_util.info().global_rank == 0:
      self.voxel_metrics_df = compute_voxel_metrics(
          self.confusion_matrix, self.classes)

  def get_mean_iou(self):
    mm = self.voxel_metrics_df
    assert mm.columns[-1] == GLOBAL_CLASS_NAME
    assert mm.columns[0] == dataset_lib.VOID_LABEL_NAME
    return float(mm.iloc[:, 1:-1].T.mean().iou)

  def write_csv(self, path: str):
    self.voxel_metrics_df.to_csv(fl := io.StringIO())
    fs.make_dirs(fs.dirname(path))
    fs.write_text(path, fl.getvalue())

  def write_tensor_board_summary(
      self, sw: torch.utils.tensorboard.SummaryWriter, global_step: int
  ):
    if not sw:
      return
    log_voxel_metrics_to_tensorboad(sw, self.voxel_metrics_df, global_step)
    sw.flush()
