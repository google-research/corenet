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

"""Evaluates a frozen CoreNet TensorFlow graph."""

import dataclasses
import logging

import torch as t

from corenet import cmd_line_flags
from corenet import configuration
from corenet import distributed as dist_util
from corenet import pipeline
from corenet import tf_model as tf_model_lib
from corenet import ui

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ProgramArgs(pipeline.DefaultProgramFlags):
  """Evaluates a frozen TensorFlow CoreNet graph."""


def main():
  dist_util.init()
  t.cuda.set_device(dist_util.info().local_rank)
  ui.initialize_logging()
  pipeline.jit_compile_cpp_modules()
  tf_model_lib.setup_tensorflow(dist_util.info().local_rank)

  args = cmd_line_flags.parse_flags(ProgramArgs)
  config, _ = pipeline.read_cmd_line_config(
      args, configuration.TfModelEvalPipeline)

  inference_fn = tf_model_lib.super_resolution_from_tf_model(
    config.frozen_graph_path)

  eval_pipe = pipeline.EvalPipeline(
      config.eval_config, inference_fn=inference_fn, tb_dir=None)
  iou = eval_pipe.run_eval(config.output_path, -1, "TF Model Eval")
  if iou is not None:
    log.info(f"Evaluation complete, mIoU={iou:.3f}")


if __name__ == '__main__':
  # @formatter:off
  # The 2 lines below provide a more structured torch tensor string
  # representation (useful when debugging).
  from corenet import debug_helpers
  debug_helpers.better_tensor_display()
  # @formatter:on

  main()
