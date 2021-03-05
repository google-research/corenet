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

"""Trains CoreNet TensorFlow graph."""
import contextlib
import dataclasses
import logging
from typing import List

import torch as t

from corenet import cmd_line_flags
from corenet import configuration
from corenet import distributed as dist_util
from corenet import file_system as fs
from corenet import misc_util
from corenet import pipeline
from corenet import state as state_lib
from corenet import super_resolution
from corenet import ui

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ProgramArgs(pipeline.DefaultProgramFlags):
  """Trains a CoreNet model."""
  recurrent_evals: bool = cmd_line_flags.flag(
      "Whether to run recurrent evals.", default=True)


class RecurrentEvals:
  @dataclasses.dataclass
  class _EvalRun:
    ev_run_eval: misc_util.StepEvent
    config: configuration.RecurrentEvalConfig
    eval_pipe: pipeline.EvalPipeline

  def __init__(self, eval_configs: List[configuration.RecurrentEvalConfig],
               state: state_lib.State, tb_root_dir: str, eval_root_dir: str):
    self.state = state
    self.eval_root_dir = eval_root_dir
    inference_fn = super_resolution.super_resolution_from_state(state)
    self.eval_runs = [
        RecurrentEvals._EvalRun(
            misc_util.StepEvent(cfg.start_step, cfg.interval),
            cfg, pipeline.EvalPipeline(
                cfg.config, inference_fn=inference_fn,
                tb_dir=fs.join(tb_root_dir, cfg.config.name)))
        for cfg in eval_configs
        if cfg.start_step >= 0
    ]

  def persistent_cpt(self, prev_step: int, next_step: int) -> bool:
    """Returns whether any eval requires a persistent checkpoint."""
    result = False
    for eval_run in self.eval_runs:
      if eval_run.ev_run_eval.trigger(prev_step, next_step):
        result = result or eval_run.config.persistent_checkpoint
    return result

  def run(self, prev_step: int, next_step: int, force=False) -> bool:
    """Runs scheduled evals and returns true if any eval has run"""
    has_run = False
    for eval_run in self.eval_runs:
      should_run = force or eval_run.ev_run_eval.trigger(prev_step, next_step)
      if not should_run:
        continue
      eval_pipe = eval_run.eval_pipe
      state = self.state
      state.model.eval()
      name = eval_pipe.config.name
      desc = f"Eval, name={name}, step={state.global_step}"
      output_dir = fs.join(self.eval_root_dir, name, f"{state.global_step:09}")
      iou = eval_pipe.run_eval(output_dir, state.global_step, desc)
      if iou is not None:
        log.info(f"Eval '{name}', step={state.global_step}, mIoU={iou:.3f}")
      has_run = True
    return has_run


def main():
  dist_util.init()
  t.cuda.set_device(dist_util.info().local_rank)
  ui.initialize_logging()
  pipeline.jit_compile_cpp_modules()

  args = cmd_line_flags.parse_flags(ProgramArgs)
  config, original_config = pipeline.read_cmd_line_config(
      args, configuration.TrainPipeline)

  output_dir = fs.normpath(fs.abspath(config.output_path))
  tb_root_dir = fs.join(output_dir, "tb")
  eval_root_dir = fs.join(output_dir, "evals")
  cpt_dir = fs.join(output_dir, "cpt")

  train_pipe = pipeline.TrainPipeline(
      config.train, cpt_dir=cpt_dir, tb_dir=fs.join(tb_root_dir, "train"))
  state = train_pipe.create_or_load_state(
      extra_metadata=original_config.to_dict())
  recurrent_evals = RecurrentEvals(config.eval, state, tb_root_dir,
                                   eval_root_dir)
  max_steps = config.train.max_steps
  train_forever = max_steps < 0
  eta = None if train_forever else misc_util.Eta(state.global_step, max_steps)
  train_pipe.switch_model_to_train()
  ev_save_temp_cpt = misc_util.StepEvent(0, config.train.checkpoint_interval)
  ev_save_pers_cpt = misc_util.StepEvent(
      0, config.train.persistent_checkpoint_interval)

  if dist_util.info().global_rank == 0:
    train_progress = ui.ProgressBar(
        desc="Training",
        bar_format=("{l_bar}{bar}| {n_fmt}/{total_fmt} "
                    "[{elapsed}, {rate_fmt}{postfix}]"),
        total=(max_steps if not train_forever else None),
        initial=state.global_step)
    bar_context = train_progress
  else:
    train_progress = None
    bar_context = contextlib.ExitStack()

  with bar_context:
    if train_progress:
      train_progress.unpause()

    while True:
      # Perform a training step
      prev_step = state.global_step
      loss = train_pipe.train_step()
      if train_progress:
        train_progress.postfix = f"loss={loss:.3f}"
        if eta:
          train_progress.postfix += f", ETA {eta.cur_eta_str(state.global_step)}"
        train_progress.update(state.global_step - train_progress.n)
      next_step = state.global_step

      should_stop = not train_forever and next_step > max_steps

      # Save a checkpoint
      if dist_util.info().global_rank == 0:
        save_pers_cpt = (
            should_stop or ev_save_pers_cpt.trigger(prev_step, next_step))
        if args.recurrent_evals:
          save_pers_cpt = (
              save_pers_cpt or
              recurrent_evals.persistent_cpt(prev_step, next_step))
        save_tmp_cpt = ev_save_temp_cpt.trigger(prev_step, next_step)

        if save_tmp_cpt or save_pers_cpt:
          train_pipe.cpt_manager.save_state(
              state_lib.encode_state(state), step=state.global_step,
              persistent=save_pers_cpt)

      # Run evaluations
      if args.recurrent_evals or should_stop:
        eval_has_run = recurrent_evals.run(
            prev_step, next_step, force=should_stop)
        if eval_has_run:
          train_pipe.switch_model_to_train()
          if train_progress:
            train_progress.unpause()

      if should_stop:
        break


if __name__ == '__main__':
  # @formatter:off
  # The 2 lines below provide a more structured torch tensor string
  # representation (useful when debugging).
  from corenet import debug_helpers
  debug_helpers.better_tensor_display()
  # @formatter:on

  main()
