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

"""High level routines for rendering voxel grids.

The API in this file has been tuned for convenient interactive usage
(e.g. in colab) rather than for use in programs.
"""

from importlib import resources
from typing import Iterable
from typing import Tuple
from typing import Union

import numpy as np
import torch as t

import corenet.misc_util as util
from corenet.geometry import transformations
from corenet.gl import rasterizer as gl
from corenet.visualization import camera_util
from corenet.visualization import shaders
from corenet.visualization.colors import DEFAULT_COLOR_PALETTE

InputTensor = Union[t.Tensor, np.ndarray, int, float, Iterable, None]


def render_voxel_grid(
    voxel_grid: InputTensor,
    view_projection_matrix: InputTensor = None,
    image_size: Tuple[int, int] = (256, 256),
    voxel_to_view_matrix: InputTensor = None,
    diffuse_coefficients: InputTensor = None,
    light_position: InputTensor = None,
    light_color: InputTensor = (1.0, 1.0, 1.0),
    ambient_light_color: InputTensor = (0.2, 0.2, 0.2),
    clear_color: Tuple[float, float, float] = (0, 0, 0),
    output_type=t.uint8,
    vertex_shader=None,
    geometry_shader=None,
    fragment_shader=None,
) -> t.Tensor:
  """Creates a voxel grid renderer py_function TF op.

  Args:
    voxel_grid: The voxel grid tensor, containing material IDs, int32[depth,
      height, width].
    view_projection_matrix: Transforms geometry from world to projected camera
      space. A float32[4, 4] row-major transformation matrix that is multiplied
      with 4 x 1 columns of homogeneous coordinates in the shader. If not
      specified, a new matrix will be calculated from the voxel grid.
    image_size: Output image shape, pair (height, width).
    voxel_to_view_matrix: Transforms from object to world space coordinates,
      float32[4, 4] transformation matrix. The voxel grid is assumed to be a
      grid of unit sized cubes placed at the origin. This matrix is responsible
      for transforming it into world space. If omitted, the voxel grid will be
      squeezed into the unit cube.
    diffuse_coefficients: The diffuse coefficients of the materials,
      float32[num_materials, 3].
    light_position: The light position, float32[3]. If None, the light will be
      placed at the camera origin.
    light_color: The light RGB color, float32[3].
    ambient_light_color: The ambient light RGB color, float32[3].
    clear_color: The RGB color to use when clearing the image, float32[3]
    output_type: The output type. Either tf.uint8 or tf.float32.
    vertex_shader: The vertex shader.
    geometry_shader: The geometry shader.
    fragment_shader: The fragment shader.

  Returns:
    The rendered image, either uint8[height, width, 3] or
    float32[height, width, 3], depending on the value of output_type.
  """

  height, width = image_size
  voxel_grid = util.to_tensor(voxel_grid, t.int32, "cpu")
  assert len(voxel_grid.shape) == 3 and voxel_grid.dtype == t.int32

  if voxel_to_view_matrix is None:
    d = 1.0 / np.max(voxel_grid.shape)
    voxel_to_view_matrix = transformations.scale([d, d, d])
  voxel_to_view_matrix = util.to_tensor(voxel_to_view_matrix, t.float32, "cpu")
  assert voxel_to_view_matrix.shape == (4, 4)

  if view_projection_matrix is None:
    mesh = t.tensor([[[0.0] * 3, [0.0] * 3, voxel_grid.shape[::-1]]])
    mesh = transformations.transform_mesh(mesh, voxel_to_view_matrix)
    view_projection_matrix = camera_util.get_default_camera_for_mesh(mesh)
  view_projection_matrix = util.to_tensor(view_projection_matrix, t.float32,
                                          "cpu")
  assert view_projection_matrix.shape == (4, 4)

  if diffuse_coefficients is None:
    diffuse_coefficients = util.to_tensor(DEFAULT_COLOR_PALETTE, t.float32,
                                          "cpu")
  diffuse_coefficients = util.to_tensor(diffuse_coefficients, t.float32, "cpu")
  assert (len(diffuse_coefficients.shape) == 2 and
          diffuse_coefficients.shape[-1] == 3)

  # By default, we use the same fragment shader as the scene renderer, which
  # needs diffuse_textures to be specified. We specify a 1x1 texture, which
  # is however not used, since we emit texture index -1 in the geometry shader.
  diffuse_textures = t.ones([1, 1, 1, 3], dtype=t.uint8)

  # The eye position in camera space is (0, 0, -1). To compute its position
  # in world space, we multiply by the inverse view-projection matrix.
  camera_position = t.mv(
      t.inverse(view_projection_matrix),
      t.tensor([0, 0, -1, 1], dtype=t.float32))
  camera_position = camera_position[:3] / camera_position[3]
  if light_position is None:
    light_position = camera_position
  light_position = util.to_tensor(light_position, t.float32, "cpu")
  assert light_position.shape == (3,)

  light_color = util.to_tensor(light_color, t.float32, "cpu")
  assert light_color.shape == (3,)

  ambient_light_color = util.to_tensor(ambient_light_color, t.float32, "cpu")
  assert ambient_light_color.shape == (3,)

  render_args = [
      gl.Uniform("voxel_to_view_matrix", voxel_to_view_matrix),
      gl.Uniform("view_projection_matrix", view_projection_matrix),
      gl.Buffer(0, voxel_grid.reshape([-1])),
      gl.Uniform("grid_resolution", voxel_grid.shape),
      gl.Buffer(1, diffuse_coefficients.reshape([-1])),
      gl.Uniform("light_position", light_position),
      gl.Uniform("camera_position", camera_position),
      gl.Texture("textures", diffuse_textures, bind_as_array=True),
      gl.Uniform("ambient_light_color", ambient_light_color),
      gl.Uniform("light_color", light_color),
  ]

  if not geometry_shader:
    geometry_shader = resources.read_text(shaders, "voxel_renderer.geom")
  if not vertex_shader:
    vertex_shader = resources.read_text(shaders, "noop.vert")
  if not fragment_shader:
    fragment_shader = resources.read_text(shaders,
                                          "point_light_illumination.frag")

  result = gl.gl_simple_render(gl.RenderInput(
      num_points=voxel_grid.numel(),
      arguments=render_args,
      output_resolution=(height, width),
      vertex_shader=vertex_shader,
      geometry_shader=geometry_shader,
      fragment_shader=fragment_shader,
      clear_color=clear_color,
      output_type=output_type
  ))
  return result[..., :3]
