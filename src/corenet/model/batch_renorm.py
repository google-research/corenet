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


class BatchRenorm(t.nn.Module):
  def __init__(self, num_channels: int, eps: float = 1e-5,
               momentum: float = 0.01):
    super().__init__()

    self.eps = eps
    self.momentum = momentum

    self.weight = t.nn.Parameter(t.ones(num_channels, dtype=t.float32))
    self.bias = t.nn.Parameter(t.zeros(num_channels, dtype=t.float32))
    self.register_buffer("running_mean", t.zeros(num_channels, dtype=t.float32))
    self.register_buffer("running_var", t.ones(num_channels, dtype=t.float32))
    self.register_buffer("num_batches_tracked", t.tensor(0, dtype=t.int64))

  # noinspection PyTypeChecker
  def forward(self, x: t.Tensor) -> t.Tensor:
    assert x.dim() >= 2
    view_dims = [1, x.shape[1]] + [1] * (x.dim() - 2)
    _v = lambda v: v.view(view_dims)
    running_std = (self.running_var + self.eps).sqrt_()  # type: t.Tensor

    if self.training:
      nt = self.num_batches_tracked  # type: t.Tensor
      d_max = (5.0 * (nt - 5000) / (25000 - 5000)).clamp_(0.0, 5.0)
      r_max = 1.0 + (2.0 * (nt - 5000) / (40000 - 5000)).clamp_(0.0, 2.0)

      reduce_dims = [i for i in range(x.dim()) if i != 1]
      b_mean = x.mean(reduce_dims)
      b_var = x.var(reduce_dims, unbiased=False)
      b_std = (b_var + self.eps).sqrt_()

      r = (b_std.detach() / running_std).clamp_(1 / r_max, r_max)
      d = ((b_mean.detach() - self.running_mean) / running_std)
      d.clamp_(-d_max, d_max)
      x = (x - _v(b_mean)) / _v(b_std) * _v(r) + _v(d)

      unbiased_var = b_var.detach() * x.shape[1] / (x.shape[1] - 1)
      self.running_var += self.momentum * (unbiased_var - self.running_var)
      self.running_mean += self.momentum * (b_mean.detach() - self.running_mean)
      self.num_batches_tracked += 1
    else:
      x = (x - _v(self.running_mean)) / _v(running_std)

    x = _v(self.weight) * x + _v(self.bias)
    return x
