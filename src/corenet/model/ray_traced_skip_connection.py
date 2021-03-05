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

"""Ray-traced skip connections."""
from typing import Callable

import torch as t
from torch import nn

import corenet.misc_util as util
from corenet.geometry import transformations


class SampleGrid2d(nn.Module):
  """Samples a 2D grid with the camera projected centers of a 3D grid."""

  def __init__(self, in_channels: int, out_channels: int,
               output_resolution: util.InputTensor):
    """Initializes the module.

    Args:
      in_channels: The number of input channels (in the 2D layer)
      out_channels: The number of output channels (in the sampled 3D grid)
      output_resolution: The 3D grid resolution (depth, height, width).
    """
    super().__init__()
    self.compress_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    grid_depth, grid_height, grid_width = output_resolution
    zz, yy, xx = t.meshgrid([
        t.arange(0, grid_depth, dtype=t.float32),
        t.arange(0, grid_height, dtype=t.float32),
        t.arange(0, grid_width, dtype=t.float32)])
    # Voxel grids are addressed using [z, y, x]
    # shape: [depth, height, width, 3]
    self.voxel_centers = t.stack([xx, yy, zz], dim=-1)

  def _apply(self, fn: Callable[[t.Tensor], t.Tensor]) -> 'SampleGrid2d':
    super()._apply(fn)
    self.voxel_centers = fn(self.voxel_centers)
    return self

  def forward(self, grid2d: t.Tensor, voxel_projection_matrix: t.Tensor,
              voxel_sample_location: t.Tensor, outside_value: float = 0,
              flip_x=False, flip_y=False):
    """The forward pass.

    Args:
      grid2d: The 2D grid, float32[batch_size, num_channels, height, width].
      voxel_projection_matrix: Matrix that projects voxel centers onto the screen,
        float32[batch_size, 4, 4].
      voxel_sample_location: 3D sample location within the voxels, float32[3].
      outside_value: Value used to fill the channels for voxels whose
        projected position is outside the 2D grid, float32[]
      flip_x: Whether to flip the 2D grid along the X dimension. This can be
        used to correct for a right/left handed 3D coordinate system issues.
      flip_y: Whether to flip the 2D grid along the Y dimension. This can be
        used to correct for a right/left handed 3D coordinate system issues.

    Returns:
      The resulting 3D grid, float32[batch_size, num_channels, depth, height,
      width]. The content of cell [b, c, z, y, x] in the result will be equal to
      grid2d[b, c, py, px], where
      (px, py, _) = affine_transform(
          voxel_projection_matrix, (x, y, z, 1)) * (height, width, 1).
      If (b, py, px) lies outside the 2D image, the content of the cell in all
      channels will be equal to outside_value.

    """
    grid2d = util.to_tensor(grid2d, t.float32)
    assert len(grid2d.shape) == 4
    voxel_sample_location = util.to_tensor(voxel_sample_location, t.float32)
    assert voxel_sample_location.shape == (grid2d.shape[0], 3)

    compressed_grid2d = self.compress_channels(grid2d)
    batch_size, channels, height, width = compressed_grid2d.shape

    voxel_projection_matrix = util.to_tensor(voxel_projection_matrix, t.float32)
    assert voxel_projection_matrix.shape == (batch_size, 4, 4)

    voxel_centers = self.voxel_centers
    grid_depth, grid_height, grid_width, _ = voxel_centers.shape
    # shape: [batch, depth, height, width, 3]
    voxel_centers = (voxel_centers[None]
                     .expand(batch_size, grid_depth, grid_height, grid_width, 3)
                     .contiguous())
    voxel_centers = (
          voxel_centers + voxel_sample_location[:, None, None, None, :])
    # shape: [batch, depth * height * width, 3]
    voxel_centers = voxel_centers.reshape([batch_size, -1, 3])

    # Project the voxel centers onto the screen
    projected_centers = transformations.transform_points_homogeneous(
        voxel_centers, voxel_projection_matrix, w=1)
    projected_centers = projected_centers.reshape([batch_size, grid_depth,
                                                   grid_height, grid_width, 4])

    camera_depth = projected_centers[..., 2]
    projected_centers = projected_centers[..., :3] / projected_centers[..., 3:4]

    # XY range in OpenGL camera space is [-1:1, -1:1]. Transform to [0:1, 0:1].
    projected_centers = projected_centers[..., :2] / 2 + 0.5

    if flip_y:
      projected_centers = projected_centers * (1, -1) + (0, 1)
    if flip_x:
      projected_centers = projected_centers * (-1, 1) + (1, 0)

    # projected_centers contains (x, y) coordinates in [0, 1]^2 at this point.
    # Convert to indices into 2D grid.
    wh = projected_centers.new_tensor([[[[[width, height]]]]], dtype=t.float32)
    pixel_indices = (projected_centers * wh).to(t.int64)
    xx, yy = pixel_indices.unbind(-1)  # type: t.Tensor
    bb = t.arange(batch_size, dtype=t.int64, device=grid2d.device)
    bb = bb[:, None, None, None]
    bb = bb.expand(batch_size, grid_depth, grid_height, grid_width)

    # Pad the grid to detect voxels which project outside the image plane
    padded_grid2d = t.constant_pad_nd(compressed_grid2d, [1, 1, 1, 1],
                                      value=outside_value)
    xx = (xx + 1).clamp(0, padded_grid2d.shape[-1] - 1)
    yy = (yy + 1).clamp(0, padded_grid2d.shape[-2] - 1)

    # Sample the 2D grid
    result = padded_grid2d[bb, :, yy, xx].permute([0, 4, 1, 2, 3])
    assert result.shape == (batch_size, channels, grid_depth, grid_height,
                            grid_width)

    # Discard voxels behind the camera
    camera_depth = camera_depth[:, None, :, :, :].expand(result.shape)
    result = t.where(camera_depth >= 0, result,
                     t.ones_like(result) * outside_value)

    return result
