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

"""Voxelization routines."""
from importlib import resources
from typing import Iterable
from typing import Tuple
from typing import Union

import numpy as np
import torch as t

import corenet.misc_util as util
from corenet.geometry import shaders
from corenet.geometry import transformations
from corenet.gl import rasterizer as gl

InputTensor = Union[t.Tensor, np.ndarray, int, float, Iterable]


def voxelize_mesh(triangles: InputTensor,
                  mesh_num_tri: InputTensor,
                  resolution: Tuple[int, int, int],
                  view2voxel: InputTensor,
                  sub_grid_sampling: bool = False,
                  image_resolution_multiplier: float = 4,
                  conservative_rasterization: bool = False,
                  projection_depth_multiplier: int = 1,
                  cuda_device=None,
                  ):
  """Voxelizes a collection of meshes.

  Voxel (x, y, z) will span the cube [x, x+1) * [y, y+1) * [z, z+1) in voxel
  space.

  Args:
    triangles: The mesh triangles, float32[total_triangles, 3, 3]
    mesh_num_tri: The number of triangles in each mesh, int32[num_meshes]
    resolution: The desired grid resolution, [depth, height, width]
    view2voxel: Transforms geometry into voxel space, float32[4, 4].
    sub_grid_sampling: Voxelize into a virtual grid and aggregate into a
      non-uniform grid. See below.
    image_resolution_multiplier: Determines the image resolution used to render
      the triangles as a function of the voxel grid resolution.
    conservative_rasterization: Whether to enable conservative rasterization.
    projection_depth_multiplier: Multiplier for the depth (see notes below).
    cuda_device: The GPU to use, given as a CUDA device index

  Returns:
    The voxelized mesh. float32[num_meshes, depth, height, width] if
    sub_grid_sampling is off,
    float32[num_meshes, 2 * depth + 1, 2 * height + 1, 2 * width + 1] otherwise.
    There is no guarantee about the device that this tensor will reside on.

  This function rasterizes each triangle using orthographic projection on one
  of the three axis aligned planes (OXY, OXZ, OYZ). For each resulting
  fragment, it marks the voxel corresponding to the fragments center in voxel
  space as occupied. To avoid holes, it chooses the plane that maximizes the
  triangle's projected surface area, similar to
  https://developer.nvidia.com/content/basics-gpu-voxelization.

  With sub-grid sampling enabled, this function voxelizes into a virtual grid
  `V` that is `image_resolution_multiplier` times larger than `resolution`. It
  then max-pools the result into an irregular voxel grid `R` with resolution
  `2 * resolution + 1`,  with the following properties:
  The center of voxel `1 + 2*(x, y, z)` in `R` corresponds to the center of
  voxel `(x, y, z)` in the voxel grid `G` defined by `world2voxel` and
  `resolution`. The cube side of voxel `1 + 2*(x, y, z)` in `R`
  is `1 / image_resolution_multiplier` and this voxel corresponds to voxel
  `(x, y, z) * image_resolution_multiplier + image_resolution_multiplier // 2`
  in `V`. Combined with empty pocket filling, `R` can be used to approximately
  determine if the voxels centers of `G` are inside the object or not.
  `image_resolution_multiplier` contains the degree of approximation.
  The return of this function with sub-grid sampling enabled is `R`.


  Notes:
    Some models in the CoreNet paper were evaluated with
    `projection_depth_multiplier=2`, others -- with `1`. The difference
    between `1` and `2` is negligible -- at 128^3 only ~0.003% of all output
    voxels differ. To reproduce the IoUs reported in the paper exactly,
    use the multipliers in the corresponding config files. If you change
    `projection_depth_multiplier` in one of these files, be sure to also change
    `image_resolution_multiplier` so the final `image_resolution` below
    remains unchanged.
  """
  triangles = util.to_tensor(triangles, dtype=t.float32)
  assert triangles.shape[1:] == (3, 3)
  mesh_num_tri = util.to_tensor(mesh_num_tri, dtype=t.int32)
  assert len(mesh_num_tri.shape) == 1
  view2voxel = util.to_tensor(view2voxel, dtype=t.float32)
  if len(view2voxel.shape) == 2:
    view2voxel = view2voxel[None].expand(len(mesh_num_tri), 4, 4)
  assert view2voxel.shape == (len(mesh_num_tri), 4, 4)

  if sub_grid_sampling and image_resolution_multiplier % 2 == 0:
    raise ValueError(
        "image_resolution_multiplier must be off if sub_grid_sampling is True")

  if sub_grid_sampling and projection_depth_multiplier == 0:
    raise ValueError(
        "projection_depth_multiplier must be 1 if sub_grid_sampling is True")

  geometry_shader = resources.read_text(shaders, "voxelize.geom")
  fragment_shader = resources.read_text(shaders, "voxelize.frag")
  vertex_shader = resources.read_text(shaders, "noop.vert")

  shape_index = util.dynamic_tile(mesh_num_tri)
  num_meshes = mesh_num_tri.shape[0]
  depth, height, width = resolution

  projection_matrix = transformations.ortho_lh(
      0, width, height, 0, 0, depth * projection_depth_multiplier)

  output_resolution = np.array([num_meshes, depth, height, width], np.int32)
  if sub_grid_sampling:
    output_resolution = output_resolution * (1, 2, 2, 2) + (0, 1, 1, 1)
  output_resolution = tuple(output_resolution)

  grid_param = gl.Buffer(
      binding=5, is_io=True,
      value=t.zeros(output_resolution, dtype=t.float32).reshape([-1]))

  render_args = [
      gl.Uniform("view_projection_matrix", projection_matrix),
      gl.Buffer(0, triangles.reshape([-1])),
      gl.Buffer(1, shape_index.reshape([-1])),
      gl.Buffer(2, view2voxel.reshape([-1])),
      gl.Uniform("voxel_grid_shape", (width, height, depth, num_meshes)),
      gl.Uniform("virtual_voxel_side",
                 image_resolution_multiplier if sub_grid_sampling else -1),
      grid_param,
  ]

  image_resolution = max(
      width, height, depth * projection_depth_multiplier)
  image_resolution = int(round(image_resolution * image_resolution_multiplier))
  _ = gl.gl_simple_render(
      gl.RenderInput(
          num_points=triangles.shape[0],
          arguments=render_args,
          output_resolution=(image_resolution,) * 2,
          vertex_shader=vertex_shader,
          geometry_shader=geometry_shader,
          fragment_shader=fragment_shader,
          clear_color=(0, 0, 0, 1),
          output_type=t.uint8,
          depth_test_enabled=False,
          conservative_rasterization=conservative_rasterization),
      cuda_device=cuda_device
  )
  grid = grid_param.value.reshape(output_resolution)
  return grid


def get_sub_grid_centers(grid: t.Tensor) -> t.Tensor:
  """Returns the occupancy at the sub-grid centers.

  Args:
    grid: Voxel grid with non-homogeneous voxels,
      float32[B, 2*D+1, 2*H+1, 2*W+1]

  Returns:
    Occupancy at the sub-grid centers, float32[B, D, H, W]

  """
  grid = grid[:, 1:, 1:, 1:]
  b, d, h, w = grid.shape
  grid = grid.reshape([b, d // 2, 2, h // 2, 2, w // 2, 2])
  grid = grid[:, :, 0, :, 0, :, 0]
  return grid
