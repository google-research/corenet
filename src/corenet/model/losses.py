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

import torch as t
import torch.nn.functional as F


def iou_agnostic(gt_volume: t.Tensor,
                 logits: t.Tensor,
                 weights: t.Tensor = None) -> t.Tensor:
  """Class-agnostic IoU like loss.

  Args:
    gt_volume: The ground truth volume containing voxel classes, int64[B,D,H,W].
    logits: The predicted voxel grid logits, float32[B,C,D,H,W].
    weights: Per-voxel loss weights, float32[B,D,H,W].

   Returns:
     The loss tensor, shape=[]
   """
  b, c, d, h, w = logits.shape
  assert logits.dtype == t.float32
  assert gt_volume.shape == (b, d, h, w) and gt_volume.dtype == t.int64

  gt_volume = F.one_hot(gt_volume, c).to(t.float32).permute([0, 4, 1, 2, 3])
  predicted_grid = logits.softmax(dim=1)

  gt_volume = gt_volume[:, 1:, ...]
  predicted_grid = predicted_grid[:, 1:, ...]

  ones_weight = gt_volume.new_tensor(c, dtype=t.float32) - 1
  ones_weight = t.ones_like(gt_volume) * ones_weight
  zeros_weight = t.ones_like(gt_volume)
  final_weights = t.where(gt_volume == 0, zeros_weight, ones_weight)
  if weights is not None:
    assert weights.shape == (b, d, h, w) and weights.dtype == t.float32
    final_weights = final_weights * weights[:, None, ...]

  intersection = t.min(gt_volume, predicted_grid) * final_weights
  union = t.max(gt_volume, predicted_grid) * final_weights

  # Compute the loss for each element in the batch separately
  intersection = intersection.sum(dim=[1, 2, 3, 4])
  union = union.sum(dim=[1, 2, 3, 4])

  iou = intersection / t.where(union == 0, t.ones_like(union), union)
  batch_iou = iou.mean()
  assert batch_iou.dtype == t.float32 and batch_iou.shape == ()

  return 1 - batch_iou


def iou_fgbg(gt_volume: t.Tensor,
             logits: t.Tensor,
             weights: t.Tensor = None) -> t.Tensor:
  """IoU like FG/BG loss.

  Collapses all foreground classes, when more than one is present. With two
  classes, the loss is equivalent to iou_agnostic.

  Args:
    gt_volume: The ground truth volume containing voxel classes, int64[B,D,H,W].
    logits: The predicted voxel grid logits, float32[B,C,D,H,W].
    weights: Per-voxel loss weights, float32[B,D,H,W].

  Returns:
    The loss tensor, shape=[]
  """
  b, c, d, h, w = logits.shape
  assert logits.dtype == t.float32
  assert gt_volume.shape == (b, d, h, w) and gt_volume.dtype == t.int64

  gt_volume = F.one_hot(gt_volume, c).to(t.float32).permute([0, 4, 1, 2, 3])

  # Logits -> class PDF
  predicted_grid = logits.softmax(dim=1)

  # Turn into FG/BG prediction problem. Use sum, as the values in the grids
  # are probabilities. Ignore class 0 (the background).
  predicted_grid = predicted_grid[:, 1:, ...].sum(1)
  gt_volume = gt_volume[:, 1:, ...].sum(1)
  # Here, predicted_grid=[B,D,H,W], gt_volume=[B,D,H,W]

  # Voxelization is not perfect, so some voxels might overlap in the
  # different objects
  gt_volume = t.min(gt_volume, gt_volume.new_tensor(1.0))

  intersection = t.min(gt_volume, predicted_grid)
  union = t.max(gt_volume, predicted_grid)

  if weights is not None:
    assert weights.shape == (b, d, h, w) and weights.dtype == t.float32
    intersection = intersection * weights
    union = union * weights

  intersection = intersection.reshape([b, -1]).sum(1)
  union = union.reshape([b, -1]).sum(1)

  iou = intersection / t.where(union == 0, t.ones_like(union), union)
  batch_iou = iou.mean()
  assert batch_iou.dtype == t.float32 and batch_iou.shape == ()

  return 1 - batch_iou


def xent(gt_volume: t.Tensor, logits: t.Tensor,
         weights: t.Tensor = None) -> t.Tensor:
  """Softmax cross entropy loss.

  Args:
    gt_volume: The ground truth volume containing voxel classes, int64[B,D,H,W].
    logits: The predicted voxel grid logits, float32[B,C,D,H,W].
    weights: Per-voxel loss weights, float32[B,D,H,W].

  Returns:
    The loss tensor, float32[]
  """
  b, c, d, h, w = logits.shape
  assert logits.dtype == t.float32
  assert gt_volume.shape == (b, d, h, w) and gt_volume.dtype == t.int64

  loss = F.cross_entropy(logits, gt_volume, reduction="none")

  if weights is not None:
    assert weights.shape == (b, d, h, w) and weights.dtype == t.float32
    loss = loss * weights

  loss = loss.mean()
  assert loss.dtype == t.float32 and loss.shape == ()
  return loss


def xent_times_iou_agnostic(gt_volume: t.Tensor,
                            predicted_grid_logits: t.Tensor,
                            weights: t.Tensor = None) -> t.Tensor:
  """Combination of IoU loss with cross entropy loss.

  Args:
    gt_volume: The ground truth volume, where voxels belonging to the object are
      1, the rest -- 0.
    predicted_grid_logits: The predicted voxel grid logits.
    weights: How much each voxel contributes to the final loss. If None, each
      voxel contributes equally. Same shape and type as gt_volume.

  Returns:
    The loss tensor, shape=[]
  """
  return ((1 + iou_agnostic(gt_volume, predicted_grid_logits, weights)) *
          (1 + xent(gt_volume, predicted_grid_logits, weights)))


def xent_times_iou_fgbg(gt_volume: t.Tensor,
                        predicted_grid_logits: t.Tensor,
                        weights: t.Tensor = None) -> t.Tensor:
  """Combination of IoU loss with cross entropy loss.

  Args:
    gt_volume: The ground truth volume, where voxels belonging to the object are
      1, the rest -- 0.
    predicted_grid_logits: The predicted voxel grid logits.
    weights: How much each voxel contributes to the final loss. If None, each
      voxel contributes equally. Same shape and type as gt_volume.

  Returns:
    The loss tensor, shape=[]
  """
  return ((1 + iou_fgbg(gt_volume, predicted_grid_logits, weights)) *
          (1 + xent(gt_volume, predicted_grid_logits, weights)))
