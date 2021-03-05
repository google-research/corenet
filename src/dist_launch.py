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

r"""Distributed launch script, compatible with PyTorch's elastic launch."""

import dataclasses
import subprocess
import sys
import time
from typing import List

import os

import corenet.cmd_line_flags as flags


@dataclasses.dataclass(frozen=True)
class ProgramArgs:
  nnodes: int = flags.flag(
      "The number of nodes to use for distributed training", default=1)
  node_rank: int = flags.flag(
      "The rank of the node for multi-node distributed training", default=0)
  nproc_per_node: int = flags.flag(
      "The number of processes to launch on each node.", default=1)
  master_addr: str = flags.flag(
      "Master node's (rank 0) IP address or hostname.", default="127.0.0.1")
  master_port: int = flags.flag(
      "Master node's (rank 0) (free-) port.", default=29500)
  num_retries: int = flags.flag(
      "How many times to retry failed jobs.", default=0)
  retry_wait_sec: int = flags.flag(
      "How long to wait before retrying a job.", default=90)
  training_script: str = flags.flag(
      "The module name of the training script to be launched.",
      arg_type=flags.POSITIONAL)
  training_script_args: List[str] = flags.flag(
      "The script arguments.", arg_type=flags.REMAINDER)


def main():
  args = flags.parse_flags(ProgramArgs)

  current_env = os.environ.copy()
  current_env["MASTER_ADDR"] = args.master_addr
  current_env["MASTER_PORT"] = str(args.master_port)
  current_env["WORLD_SIZE"] = str(args.nproc_per_node * args.nnodes)
  current_env["LOCAL_WORLD_SIZE"] = str(args.nproc_per_node)
  current_env["GROUP_RANK"] = str(args.node_rank)

  if 'OMP_NUM_THREADS' not in os.environ and args.nproc_per_node > 1:
    current_env["OMP_NUM_THREADS"] = str(1)

  cmd = [sys.executable, "-u", "-m", args.training_script]
  cmd.extend(args.training_script_args)

  for cur_try in range(args.num_retries + 1):
    processes = []
    for local_rank in range(0, args.nproc_per_node):
      dist_rank = args.nproc_per_node * args.node_rank + local_rank
      current_env["RANK"] = str(dist_rank)
      current_env["LOCAL_RANK"] = str(local_rank)
      process = subprocess.Popen(cmd, env=current_env)
      processes.append(process)

    while True:
      any_process_running = False
      any_process_has_errors = False
      for process in processes:
        ret_code = process.poll()
        if ret_code is None:
          any_process_running = True
        else:
          if ret_code != 0:
            any_process_has_errors = True

      if any_process_has_errors:
        completed_successfully = False
        break
      if not any_process_running:
        completed_successfully = True
        break
      time.sleep(1)

    if completed_successfully:
      break
    else:
      for process in processes:
        process.kill()
        process.wait()
      if cur_try < args.num_retries:
        print(f"Job failed, attempt={cur_try + 1}. Waiting and restarting...")
        time.sleep(args.retry_wait_sec)
      else:
        raise ValueError(f"Job failed, all retry attempts exhausted")


if __name__ == "__main__":
  main()
