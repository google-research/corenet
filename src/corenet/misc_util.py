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

"""Miscellaneous utilities that don't fit anywhere else."""

import dataclasses
import datetime
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import TypeVar
from typing import Union

import numpy as np
import torch as t
import torch.distributed

InputTensor = Union[t.Tensor, np.ndarray, int, float, Iterable]


def dynamic_tile(partition_lengths: t.Tensor):
  """Computes dynamic tiling with the given partition lengths.

  Args:
    partition_lengths: The partition lengths, int32[num_partitions]

  Returns:
    A 1D int32 tensor, containing  partition_lengths[0] zeros,
    followed by partition_lengths[1] ones, followed by
    partition_lengths[2] twos, and so on.
  """
  start_index = partition_lengths.cumsum(0)
  start_index, num_elements = start_index[:-1], start_index[-1]
  result = partition_lengths.new_zeros([num_elements.item()], dtype=t.int32)
  result[start_index] = 1
  result = result.cumsum(0, dtype=t.int32)
  return result


def to_tensor(v: InputTensor, dtype: t.dtype,
              device: Optional[Union[t.device, str]] = None) -> t.Tensor:
  """Converts a value to tensor, checking the type.

  Args:
    v: The value to convert. If it is already a tensor or an array, this
      function checks that the type is equal to dtype. Otherwise, uses
      torch.as_tensor to convert it to tensor.
    dtype: The required type.
    device: The target tensor device (optional(.

  Returns:
    The resulting tensor

  """
  if not t.is_tensor(v):
    if hasattr(v, "__array_interface__"):
      # Preserve the types of arrays. The must match the given type.
      v = t.as_tensor(v)
    else:
      v = t.as_tensor(v, dtype=dtype)

  if v.dtype != dtype:
    raise ValueError(f"Expecting type '{dtype}', found '{v.dtype}'")

  if device is not None:
    v = v.to(device)

  return v


def safe_div(x: t.Tensor, y: t.Tensor) -> t.Tensor:
  """Returns x/y where y != 0 and 0 where y = 0"""
  return t.where(y != 0, x / y, y)


TorchDevice = Union[t.device, str, None]

T = TypeVar('T')


class TensorContainerMixin:
  """Allows unified operation on all tensors contained in a dataclass."""

  def _apply(self, fn: Callable[[t.Tensor], t.Tensor]):
    result = []
    for field in dataclasses.astuple(self):
      if t.is_tensor(field):
        field = fn(field)
      elif isinstance(field, list) or isinstance(field, tuple):
        field = [fn(e) if t.is_tensor(e) else e for e in field]
      result.append(field)
    return type(self)(*result)

  def cuda(self: T) -> T:
    return self._apply(lambda v: v.cuda())

  def cpu(self: T) -> T:
    return self._apply(lambda v: v.cpu())

  def numpy(self: T) -> T:
    return self._apply(lambda v: v.numpy())

  def to(self: T, device: TorchDevice) -> T:
    return self._apply(lambda v: v.to(device))

  def reduce_multigpu(
      self: T, dst: int,
      op: t.distributed.reduce_op = t.distributed.ReduceOp.SUM
  ) -> T:
    result = t.distributed.reduce_multigpu(dataclasses.astuple(self), dst, op)
    return type(self)(*result)


def round_up(n: int, div: int):
  """Round up to the nearest multiplier of div."""
  return ((n + div - 1) // div) * div


class TimedEvent:
  """Helper class that spaces successive event executions in time."""

  def __init__(self, interval_sec: float):
    """Initializes the class.

    Args:
      interval_sec: The minimum interval between two successive executions.
    """
    self.last_trigger_time = datetime.datetime.min
    self.interval_sec = interval_sec

  def trigger(self) -> bool:
    """Returns `True` if at least interval_sec have passed from last `True`."""
    cur_time = datetime.datetime.now()
    time_delta = (cur_time - self.last_trigger_time).total_seconds()
    if time_delta > self.interval_sec:
      self.last_trigger_time = cur_time
      return True
    return False

  def __repr__(self):
    return f"{type(self).__name__}@{self.interval_sec}s"


class StepEvent:
  """Helper class for semi-regular event execution."""

  def __init__(self, start_step: int, interval: int):
    self.start_step = start_step
    self.interval = interval

  def trigger(self, prev_step: int, next_step: int):
    """Returns True if at least one execution is between prev_ and cur_ step.

    In more detail, executes if there is an integer K >= 0 such that:
    prev_step < start_step + K * interval + 0.5 < next_step
    """
    if next_step < self.start_step:
      return False
    if prev_step < self.start_step:
      return True
    prev_rep = (prev_step - self.start_step) // self.interval
    cur_rep = (next_step - self.start_step) // self.interval
    return prev_rep != cur_rep

  def __repr__(self):
    return f"{type(self).__name__}@{self.start_step}+{self.interval}N"


class Eta:
  def __init__(self, start: int, total: int):
    self.start = start
    self.total = total
    self.start_time = datetime.datetime.now()

  def cur_eta_sec(self, current: int):
    cur_time = datetime.datetime.now()
    sec_per_it = (
        (cur_time - self.start_time).total_seconds() / (current - self.start))
    return sec_per_it * (self.total - current)

  def cur_eta_str(self, current: int):
    s = int(self.cur_eta_sec(current))
    return f"{s // 86400}d:{s % 86400 // 3600}h:{s % 3600 // 60}m:{s % (60)}s"
