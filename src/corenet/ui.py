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

"""Utilities for displaying progress and messages."""
import datetime
import logging
import os.path
import sys
import time
from logging import LogRecord
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import TypeVar
from typing import Union

import os
import torch as t
import torch.distributed
import torch.distributed.rpc
import tqdm

from corenet import distributed as dist_util
from corenet import misc_util


class ProgressBar(tqdm.std.tqdm):
  """A tqdm based progress bar that works correctly with redirected output."""

  def __init__(self, *args, **kwargs):
    file = kwargs.get("file", sys.stderr)
    self.is_atty = hasattr(file, "isatty") and file.isatty()
    force_atty = int(os.environ.get("FORCE_IS_ATTY", "-1"))
    if force_atty >= 0:
      self.is_atty = bool(force_atty)

    force_update_sec = float(os.environ.get("FORCE_NOATTY_UPDATE_SEC", "-1"))
    self.ev_update_display = misc_util.TimedEvent(
        force_update_sec if force_update_sec > 0 else 20)
    self.__closing = False

    kwargs = dict(kwargs)
    if not self.is_atty:
      kwargs["leave"] = True
      kwargs["ncols"] = min(240, kwargs.get("ncols", 160))

    super().__init__(*args, **kwargs)

  def status_printer(self, fp):
    fp_flush = getattr(fp, 'flush', lambda: None)
    last_len = [0]

    def fp_write(s):
      fp.write(str(s))
      fp_flush()

    def print_status(s):
      if self.is_atty:
        len_s = len(s)
        fp_write('\r' + s + (' ' * max(last_len[0] - len_s, 0)))
        last_len[0] = len_s
      elif self.__closing or self.ev_update_display.trigger():
        fp_write(s + "\n")

    return print_status

  def moveto(self, n):
    if self.is_atty:
      super().moveto(n)

  def close(self):
    self.__closing = True
    super().close()


def write(s: str, file=None, end="\n"):
  """Writes on the console in a progress bar compatible way."""
  ProgressBar.write(s, file, end)


T = TypeVar("T")

# Arguments are (current_progress: int, max_progress: int, worker_status: str)
DistributedProgressReportFn = Callable[[int, int, str], None]


def progress_bar_report_fn(
    bar: ProgressBar, progress_multiplier: Union[float, int] = 1
) -> Optional[DistributedProgressReportFn]:
  """Returns a function that reports progress using a progress bar."""

  if bar is None:
    return None

  def display(current: int, total: int, worker_status: str):
    bar.n = current * progress_multiplier
    bar.total = total * progress_multiplier
    bar.postfix = f"W=|{worker_status}|"
    bar.display()

  return display


class DistributedProgress:
  """Distributed async progress reporter.

  Example usage:
    progress = DistributedProgress(dist_info)
    for v in progress(iterable):
      pass

  """

  _max_progress: Optional[t.Tensor] = None
  _current_progress: Optional[t.Tensor] = None

  @staticmethod
  def _create_structs():
    dist_info = dist_util.info()
    cls = DistributedProgress
    if dist_info.global_rank == 0:
      if cls._max_progress is not None:
        raise ValueError("Multiple distributed progress bars not supported!")
      world_size = dist_info.global_world_size
      cls._max_progress = t.ones(world_size, dtype=t.int64, device="cpu") * -1
      cls._current_progress = t.zeros(world_size, dtype=t.int64, device="cpu")
    t.distributed.barrier()

  @staticmethod
  def _set_max_progress(rank: int, max_progress: int):
    DistributedProgress._max_progress[rank] = max_progress

  @staticmethod
  def _update_progress(rank: int, current_progress: int):
    DistributedProgress._current_progress[rank] = current_progress

  def __init__(
      self,
      report_progress_fn: Optional[DistributedProgressReportFn] = None
  ):
    """Initializes the progress reporter.

    Args:
      report_progress_fn: Called to report progress. Arguments are
        (current_progress: int, max_progress: int, worker_status: str). If not
        specified, will use a ProgressBar() bar to report progress.
    """
    if dist_util.info().global_rank == 0:
      if report_progress_fn is None:
        report_progress_fn = progress_bar_report_fn(ProgressBar())
      self.report_progress_fn = report_progress_fn
    elif report_progress_fn is not None:
      raise ValueError("Only rank 0 can specify a progress reporting function.")

  def __call__(self, seq: Sequence[T]) -> Iterable[T]:
    ev_rpc_report = misc_util.TimedEvent(1)
    ev_call_report_fn = misc_util.TimedEvent(0.2)

    rank = dist_util.info().global_rank
    master_node = dist_util.get_node_name(0)
    seq_len = len(seq)
    C = DistributedProgress
    self._create_structs()
    t.distributed.rpc.rpc_sync(master_node, C._set_max_progress,
                               (rank, seq_len))
    for progress, v in enumerate(seq):
      if ev_rpc_report.trigger():
        t.distributed.rpc.rpc_sync(master_node, C._update_progress,
                                   (rank, progress))
      if rank == 0 and ev_call_report_fn.trigger():
        self._report_progress()
      yield v
    t.distributed.rpc.rpc_sync(master_node, C._update_progress, (rank, seq_len))
    async_op = t.distributed.barrier(async_op=True)
    if rank == 0:
      self._flush(async_op)
    else:
      async_op.wait()

  def _flush(self, async_op):
    """Waits workers to finish, while reporting progress."""
    while True:
      self._report_progress()
      if async_op.is_completed():
        break
      time.sleep(0.2)
    DistributedProgress._max_progress = None
    DistributedProgress._current_progress = None

  def _report_progress(self):
    max_progress = DistributedProgress._max_progress
    current_progress = DistributedProgress._current_progress
    progress_bars = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    worker_status = "".join(
        f"{progress_bars[min(max(int(c / m * 8), 0), 7)]}" if m >= 0 else "?"
        for c, m in t.stack([current_progress, max_progress], -1))
    current = int(current_progress.sum())
    total = int(t.max(max_progress, t.zeros_like(max_progress)).sum())
    self.report_progress_fn(current, total, worker_status)


class TqdmLoggingHandler(logging.Handler):
  def __init__(self, level=logging.NOTSET):
    super().__init__(level)

  @staticmethod
  def _write_to_tqdm(msg: str):
    write(msg)

  def emit(self, record):
    try:
      msg = self.format(record)
      t.distributed.rpc.rpc_async(
          dist_util.get_node_name(0), TqdmLoggingHandler._write_to_tqdm, (msg,))
      self.flush()
    except (KeyboardInterrupt, SystemExit):
      raise
    except:
      self.handleError(record)


class LoggingFormater(logging.Formatter):

  def __init__(self):
    super().__init__(fmt="{message}", style="{")

  def format(self, record: LogRecord) -> str:
    msg = super().format(record)
    rank = t.distributed.get_rank() if t.distributed.is_initialized() else "N/A"
    fname = os.path.basename(record.filename)
    d = datetime.datetime.fromtimestamp(record.created)
    msg = (f"{record.levelname[0]}{d:%y%m%d %H:%M:%S.%f} W:{rank} "
           f"{fname}:{record.lineno}] {msg}")

    return msg


def initialize_logging():
  logging.root.setLevel(logging.INFO)
  handler = TqdmLoggingHandler()
  handler.setFormatter(LoggingFormater())
  logging.root.addHandler(handler)
  logging.root.info("Initialized logging")
