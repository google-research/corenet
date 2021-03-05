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
import logging
from typing import Tuple

import torch as t

from corenet import pipeline
from corenet import state as state_lib
from corenet.geometry import transformations

log = logging.getLogger(__name__)


class MultiOffsetInferenceFn:
  def __call__(
      self, input_image: t.Tensor, camera_transform: t.Tensor,
      view_to_voxel_transform: t.Tensor, grid_offsets: t.Tensor
  ) -> t.Tensor:
    """An inference function that can operate on multiple offsets.
    Args:
      input_image: float32[batch_size, 3, height, width]
      camera_transform: float32[batch_size, 4, 4]
      view_to_voxel_transform: float32[batch_size, 4, 4]
      grid_offsets: float32[num_offsets, batch_size, 3]
    Returns:
      pmf: float32[num_offsets, batch_size, num_classes, depth, height, width]
        multinomial distribution over the possible classes
    """
    raise NotImplementedError()


class SuperResolutionInference(pipeline.InferenceFn):
  def __init__(self, inference_fn: MultiOffsetInferenceFn,
               resolution: Tuple[int, int, int]):
    self.resolution = resolution
    self.inference_fn = inference_fn
    self.offset_cache = {}

  def get_resolution_multiplier(self, output_resolution: Tuple[int, int, int]):
    """Returns the multiplier between the native and the output resolutions."""
    resolution_multiplier = (
        t.as_tensor(output_resolution, dtype=t.float32) /
        t.as_tensor(self.resolution, dtype=t.float32))
    if ((resolution_multiplier.floor() != resolution_multiplier.ceil()).any() or
        (resolution_multiplier < 1).any() or
        resolution_multiplier.min() != resolution_multiplier.max()):
      raise ValueError(
          "The output resolution should be divisible by the native resolution")
    resolution_multiplier = int(resolution_multiplier[0])
    return resolution_multiplier

  def get_native_offsets(self, output_resolution: Tuple[int, int, int],
                         grid_offsets: t.Tensor) -> t.Tensor:
    """Returns the sampling offsets in the native grid.

    Args:
      output_resolution: The output grid resolution, (depth, height, width).
      grid_offsets: The offsets in the output grid, float32[batch_size, 3]

    Returns:
      The sampling offsets in the native grid,
        float32[resolution_multiplier**3, batch_size, 3]

    """
    output_resolution = tuple(output_resolution)
    assert len(output_resolution) == 3
    resolution_multiplier = self.get_resolution_multiplier(output_resolution)
    if output_resolution not in self.offset_cache:
      zz, yy, xx = t.meshgrid(
          [t.arange(resolution_multiplier, device="cpu")] * 3)
      offsets = t.stack([xx, yy, zz], -1) / resolution_multiplier
      offsets = offsets.reshape([-1, 3])
      self.offset_cache[output_resolution] = offsets
    offsets = self.offset_cache[output_resolution].to(grid_offsets.device)
    offsets = offsets[:, None] + grid_offsets[None, :] / resolution_multiplier
    return offsets

  def __call__(
      self, input_image: t.Tensor, camera_transform: t.Tensor,
      view_to_voxel_transform: t.Tensor, grid_offsets: t.Tensor,
      output_resolution: Tuple[int, int, int]
  ) -> t.Tensor:
    native_offsets = self.get_native_offsets(output_resolution, grid_offsets)
    resolution_multiplier = self.get_resolution_multiplier(output_resolution)

    batch_size = input_image.shape[0]
    md, mh, mw = [resolution_multiplier] * 3
    scale = transformations.scale([1 / md, 1 / mh, 1 / mw])
    view_to_voxel_transform = (
        view_to_voxel_transform @ scale.to(view_to_voxel_transform.device))
    pmfs = self.inference_fn(input_image, camera_transform,
                             view_to_voxel_transform, native_offsets)

    _, _, num_channels, d, h, w = pmfs.shape
    pmfs = pmfs.reshape([md, mh, mw, batch_size, num_channels, d, h, w])
    pmfs = pmfs.permute([3, 4, 5, 0, 6, 1, 7, 2])
    pmfs = pmfs.reshape([batch_size, num_channels, md * d, mh * h, mw * w])
    return pmfs


def super_resolution_from_state(
    state: state_lib.State) -> SuperResolutionInference:
  def inference_fn(
      input_image: t.Tensor, camera_transform: t.Tensor,
      view_to_voxel_transform: t.Tensor, grid_offsets: t.Tensor
  ):
    v2s = camera_transform @ view_to_voxel_transform.inverse()
    pmfs = []
    for grid_offset in grid_offsets:
      logits = state.model(input_image, v2s, grid_offset)
      pmfs.append(logits.softmax(dim=1))
    return t.stack(pmfs, dim=0)

  native_resolution = state.model.config.decoder.resolution
  return SuperResolutionInference(inference_fn, native_resolution)
