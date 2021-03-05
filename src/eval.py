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

import dataclasses
import logging

import re
import torch as t

from corenet import cmd_line_flags
from corenet import configuration as config_lib
from corenet import distributed as dist_util
from corenet import file_system as fs
from corenet import pipeline
from corenet import state as state_lib
from corenet import super_resolution
from corenet import ui

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ProgramArgs(pipeline.ConfigPostProcessFlags):
  """Evaluates a CoreNet model."""
  cpt_path: str = cmd_line_flags.flag(
      "Path to the CoreNet checkpoint.", default=None)
  output_path: str = cmd_line_flags.flag("Output directory.", default=None)
  eval_names_regex: str = cmd_line_flags.flag(
      "Regex for the evaluations to run", default=".*")


def main():
  dist_util.init()
  t.cuda.set_device(dist_util.info().local_rank)
  ui.initialize_logging()
  pipeline.jit_compile_cpp_modules()

  args = cmd_line_flags.parse_flags(ProgramArgs)

  # Read the state and create an inference function
  raw_state = fs.read_bytes(args.cpt_path)
  state = state_lib.decode_state(
      raw_state, f"cuda:{dist_util.info().local_rank}")
  state.model.eval()
  inference_fn = super_resolution.super_resolution_from_state(state)

  # Read and post-process the configuration
  train_pipe_config = config_lib.TrainPipeline.from_dict(state.extra_metadata)
  train_pipe_config = pipeline.post_process_config(train_pipe_config, args)

  # Run the evals
  eval_root_dir = args.output_path
  for eval_config in train_pipe_config.eval:
    eval_config = eval_config.config
    if not re.match(args.eval_names_regex, eval_config.name):
      continue
    eval_pipe = pipeline.EvalPipeline(
        eval_config, inference_fn=inference_fn, tb_dir=None)
    name = eval_pipe.config.name
    desc = f"Eval, name={name}, step={state.global_step}"
    eval_dir = fs.join(eval_root_dir, eval_config.name)
    iou = eval_pipe.run_eval(eval_dir, state.global_step, desc)
    if iou is not None:
      log.info(f"Eval '{name}', step={state.global_step}, mIoU={iou:.3f}")

  dist_util.shutdown()


if __name__ == '__main__':
  # @formatter:off
  # The 2 lines below provide a more structured torch tensor string
  # representation (useful when debugging).
  from corenet import debug_helpers
  debug_helpers.better_tensor_display()
  # @formatter:on

  main()
