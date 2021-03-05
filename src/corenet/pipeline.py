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

"""Routines for building training and evaluation pipelines."""
import contextlib
import dataclasses
import logging
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar

import jq
import json5
import torch as t
import torch.utils.data
import torch.utils.tensorboard

from corenet import cmd_line_flags
from corenet import configuration
from corenet import cpt_manager as cpt_manager_lib
from corenet import distributed as dist_util
from corenet import evaluation_results as eval_results_lib
from corenet import file_system as fs
from corenet import misc_util
from corenet import state as state_lib
from corenet import ui
from corenet.cc import fill_voxels
from corenet.data import batched_example
from corenet.data import dataset as dataset_lib
from corenet.data.dataset_manager import DatasetManager
from corenet.model import losses

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ConfigPostProcessFlags:
  jq_transform: List[str] = cmd_line_flags.flag(
      "Allows to change the configuration from the command line using jq "
      "transformations. Applied before string template substitution.",
      short_name="jq")
  string_def: List[str] = cmd_line_flags.flag(
      "Allows to replace the string template in the configuration file "
      "from the command line. Format is <key>=<value>.", short_name="D")


@dataclasses.dataclass(frozen=True)
class DefaultProgramFlags(ConfigPostProcessFlags):
  config_path: str = cmd_line_flags.flag("Path to config json.")


TConfig = TypeVar('TConfig', covariant=True)


def post_process_config(config: TConfig,
                        args: ConfigPostProcessFlags) -> TConfig:
  """Post-processes a configuration according to the given flags."""
  config_type = type(config)
  config = config.to_dict()
  for jq_transform in args.jq_transform:
    config = jq.compile(jq_transform).input(config).first()
  config = config_type.from_dict(config)
  str_template_subs = configuration.parse_template_mapping(args.string_def)
  config = configuration.replace_templates(config, str_template_subs)
  return config


def read_cmd_line_config(
    args: DefaultProgramFlags, config_type: Type[TConfig]
) -> Tuple[TConfig, TConfig]:
  """Reads the configuration json and replaces string defs from command line."""
  config = config_type.from_dict(json5.loads(fs.read_text(args.config_path)))
  original_config = config
  config = post_process_config(config, args)
  return config, original_config


def jit_compile_cpp_modules():
  if dist_util.info().local_rank == 0:
    log.info("JIT compiling C++ modules... "
             "(this might take a few minutes on the first run)")
    fill_voxels.get_module()
    log.info("Done JIT compiling")
  t.distributed.barrier(dist_util.local_nccl_group())


def create_distributed_loader(
    dataset: dataset_lib.CoReNetDataset,
    loader_config: configuration.DataLoaderConfig,
    pad_data=False
) -> t.utils.data.DataLoader:
  """Creates a distributed-aware data loader for a dataset."""
  dist_info = dist_util.info()
  sampler = dist_util.DistributedSampler(
      dataset, global_rank=dist_info.global_rank,
      global_world_size=dist_info.global_world_size, pad_data=pad_data)
  ctx = (t.multiprocessing.get_context("fork")
         if loader_config.num_data_workers > 0 else None)

  # noinspection PyArgumentList
  loader_config = t.utils.data.DataLoader(
      dataset, batch_size=loader_config.batch_size,
      num_workers=loader_config.num_data_workers, pin_memory=True,
      collate_fn=lambda v: v, sampler=sampler,
      multiprocessing_context=ctx, drop_last=False,
      prefetch_factor=loader_config.prefetch_factor)

  return loader_config


def voxelize_batch(b: batched_example.BatchedExample,
                   voxelization_config: configuration.VoxelizationConfig):
  """Voxelizes a batched example with the given settings."""
  with t.no_grad():
    voxel_content_fn = {
        configuration.TaskType.SEMANTIC:
          batched_example.VoxelContentSemanticLabel(b.mesh_labels),
        configuration.TaskType.FG_BG: batched_example.voxel_content_1
    }[voxelization_config.task_type]
    resolution = dataclasses.astuple(voxelization_config.resolution)
    res_mul = voxelization_config.voxelization_image_resolution_multiplier
    res_cons = voxelization_config.conservative_rasterization
    depth_mul = voxelization_config.voxelization_projection_depth_multiplier
    b = batched_example.voxelize(
        b, resolution=resolution, voxel_content_fn=voxel_content_fn,
        sub_grid_sampling=voxelization_config.sub_grid_sampling,
        image_resolution_multiplier=res_mul,
        conservative_rasterization=res_cons,
        projection_depth_multiplier=depth_mul)

  return b


class TrainPipeline:
  def __init__(self, config: configuration.TrainConfig,
               cpt_dir: str, tb_dir: str):
    self.config = config

    loss_fns = {
        configuration.TaskType.FG_BG: losses.iou_fgbg,
        configuration.TaskType.SEMANTIC: losses.xent_times_iou_agnostic
    }
    self.loss_fn = loss_fns[config.data.voxelization_config.task_type]

    dist_info = dist_util.info()
    if dist_info.global_rank == 0:
      self.tb_writer = t.utils.tensorboard.SummaryWriter(tb_dir)
      self.ev_log_to_tb = misc_util.StepEvent(
          0, config.tensorboard_log_interval)

    self.data_manager = DatasetManager(config.data)
    self.step_size = (dist_info.global_world_size *
                      self.config.data.data_loader.batch_size)
    self.cpt_dir = cpt_dir

    self.ddp_model: Optional[t.nn.parallel.DistributedDataParallel] = None
    self._step_it = None
    self._state: Optional[state_lib.State] = None
    self.cpt_manager: Optional[cpt_manager_lib.CheckpointManager] = None

  def create_or_load_state(self, extra_metadata: Any) -> state_lib.State:
    dist_info = dist_util.info()
    if dist_info.global_rank == 0:
      self.cpt_manager = cpt_manager_lib.CheckpointManager(self.cpt_dir)
      if not self.cpt_manager.has_checkpoints():
        log.info("Initializing training from scratch")
        state = state_lib.create_initial_state(
            self.config, len(self.data_manager.classes))
        state = dataclasses.replace(state, extra_metadata=extra_metadata)
        self.cpt_manager.save_state(
            state_lib.encode_state(state), step=0, persistent=True)
      cpt_reader = self.cpt_manager
    t.distributed.barrier()
    if dist_info.global_rank != 0:
      self.cpt_manager = None
      cpt_reader = cpt_manager_lib.CheckpointReader(self.cpt_dir)

    # noinspection PyUnboundLocalVariable
    raw_state = cpt_reader.read_last_checkpoint()
    self._state = state_lib.decode_state(
        raw_state, f"cuda:{dist_info.local_rank}")
    log.info(f"Starting training from step={self._state.global_step}")

    self.ddp_model = t.nn.parallel.DistributedDataParallel(
        self._state.model, device_ids=[dist_info.local_rank])

    return self._state

  def switch_model_to_train(self):
    self._state.model.train()
    self.ddp_model.train()

  # noinspection PyUnusedLocal
  def _work_around_oom_in_ddp(self):
    """Avoid OOMs in DDP by pre-caching large blocks of memory."""
    mem_chunks = [t.zeros([int(3 * 1024 ** 3)], dtype=t.uint8,
                          device=f"cuda:{t.cuda.current_device()}")
                  for _ in range(3)]

  def _process_batch(self, batch: List[dataset_lib.DatasetElement]) -> float:
    """Processes one batch and returns the loss."""
    # Keep the batch on the CPU to reduce memory fragmentation
    batch = batched_example.batch(batch)
    batch = voxelize_batch(batch, self.config.data.voxelization_config)
    v2s = batch.camera_transform @ batch.v2x_transform.inverse()

    state = self._state
    state.optimizer.zero_grad()
    logits = self.ddp_model(
        batch.input_image.cuda(), v2s.cuda(),
        batch.grid_sampling_offset.cuda())
    assert batch.grid.device.type == "cuda"
    loss = self.loss_fn(batch.grid.to(t.int64), logits)
    loss.backward()
    state.optimizer.step()

    prev_step = state.global_step
    state.global_step += self.step_size

    cpu_loss = float(loss.detach().cpu().item())
    if dist_util.info().global_rank == 0:
      if self.ev_log_to_tb.trigger(prev_step, state.global_step):
        self.tb_writer.add_scalar("loss", cpu_loss, state.global_step)
        self.tb_writer.flush()
    return cpu_loss

  def _train_step_impl(self) -> Iterable[float]:
    """Performs a training step and returns the loss."""

    self._work_around_oom_in_ddp()
    while True:
      dataset = self.data_manager.create_dataset_from_start_step(
          self._state.global_step)
      data_loader = create_distributed_loader(
          dataset=dataset, loader_config=self.config.data.data_loader,
          pad_data=True)
      for batch in data_loader:
        yield self._process_batch(batch)

  def train_step(self) -> float:
    if not self._step_it:
      self._step_it = iter(self._train_step_impl())
    return next(self._step_it)


class InferenceFn:
  def __call__(self, input_image: t.Tensor, camera_transform: t.Tensor,
               view_to_voxel_transform: t.Tensor, grid_offsets: t.Tensor,
               output_resolution: Tuple[int, int, int]):
    """The inference function.
    Args:
      input_image: float32[batch_size, 3, img_height, img_width]
      camera_transform: float32[batch_size, 4, 4]
      view_to_voxel_transform: float32[batch_size, 4, 4]
      grid_offsets: float32[batch_size, 3]
      output_resolution: Tuple[int, int, int]
    Returns:
      pmf: float32[batch_size, num_classes, depth, height, width]
        multinomial distribution over the possible classes
    """
    raise NotImplementedError()


class EvalPipeline:
  def __init__(self, config: configuration.EvalConfig,
               inference_fn: InferenceFn, tb_dir: Optional[str]):
    self.config = config

    if dist_util.info().global_rank == 0 and tb_dir:
      self.tb_writer = t.utils.tensorboard.SummaryWriter(tb_dir)
    else:
      self.tb_writer = None
    self.data_manager = DatasetManager(config.data, global_seed=0x4F1A2379)

    self.inference_fn = inference_fn

  def run_eval(self, output_dir: str, global_step: int, progress_bar_desc: str):
    exit_stack = contextlib.ExitStack()
    if dist_util.info().global_rank == 0:
      progress_bar = ui.ProgressBar(desc=progress_bar_desc, leave=False)
      exit_stack.push(progress_bar)
    else:
      progress_bar = None
    with exit_stack:
      dataset = self.data_manager.create_dataset(local_seed=global_step)
      loader_config = self.config.data.data_loader
      data_loader = create_distributed_loader(
          dataset=dataset, loader_config=loader_config, pad_data=False)

      # Main evaluation loop
      progress_report_fn = ui.progress_bar_report_fn(
          progress_bar, progress_multiplier=loader_config.batch_size)
      progress = ui.DistributedProgress(report_progress_fn=progress_report_fn)
      qualitative_results = eval_results_lib.QualitativeResults(
          self.config, dataset, output_dir)
      quantitative_results = eval_results_lib.QuantitativeResults(
          dataset.classes, self.config)
      voxel_config = self.config.data.voxelization_config
      data_resolution = dataclasses.astuple(voxel_config.resolution)

      for batch in progress(data_loader):
        batch = batched_example.batch([v.cuda() for v in batch])
        batch = voxelize_batch(batch, voxel_config)

        with t.no_grad():
          pmf = self.inference_fn(
              batch.input_image, batch.camera_transform, batch.v2x_transform,
              batch.grid_sampling_offset, data_resolution)
        quantitative_results.add_batch(pmf, batch)
        qualitative_results.add_batch(pmf, batch)

      quantitative_results.compute_metrics()
      if dist_util.info().global_rank == 0:
        voxel_metrics_path = fs.join(output_dir, "voxel_metrics.csv")
        quantitative_results.write_csv(voxel_metrics_path)
        quantitative_results.write_tensor_board_summary(
            self.tb_writer, global_step)

      log.debug("Writing results to disk...")
      qualitative_results.write_tensor_board_summary(
          self.tb_writer, global_step)
      log.debug("Finished evaluating")
      t.distributed.barrier()

      if dist_util.info().global_rank == 0:
        return quantitative_results.get_mean_iou()
      else:
        return None
