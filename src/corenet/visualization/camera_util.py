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

"""Functions for computing camera matrices."""

import math
from typing import Iterable
from typing import Union

import numpy as np
import torch as t

import corenet.misc_util as util
from corenet.geometry import transformations

InputTensor = Union[t.Tensor, np.ndarray, int, float, Iterable]


def cameras_on_tetrahedron_vertices() -> t.Tensor:
  """Computes view matrices of cameras placed at tetrahedron vertices.

  Returns:
    The 4x4 camera transformation matrices, float32[4, 4, 4]. The first three
    cameras are above the coordinate system origin and look at the origin, while
    the last camera looks at the origin from below.

  Assumes left handed coordinate system where Y points up.
  """

  tetrahedron_vertices = np.array(
      [(math.sqrt(8.0 / 9), 1.0 / 3, 0),
       (-math.sqrt(2.0 / 9), 1.0 / 3, math.sqrt(2.0 / 3)),
       (-math.sqrt(2.0 / 9), 1.0 / 3, -math.sqrt(2.0 / 3)), (0, -1, 0)],
      dtype=np.float32)
  up_vectors = t.tensor([[0, 1, 0]] * 3 + [[1, -1, 0]], dtype=t.float32)
  matrices = []

  for camera_origin, up_vector in zip(tetrahedron_vertices, up_vectors):
    look_at = transformations.look_at_lh(camera_origin, np.zeros(3, np.float32),
                                         up_vector)
    matrices.append(look_at)
  return t.stack(matrices, 0)


def frontal_camera(offset: float) -> t.Tensor:
  """Computes view matrix of camera looking at the origin along the Z direction.

  Args:
    offset: The offset from the coordinate system origin (along Z).

  Returns:
    The camera matrix, float32[4, 4]

  Assumes left handed coordinate system where Y points up.
  """
  return transformations.look_at_lh((0, 0, offset), (0, 0, 0), (0, 11, 0))


def perspective_projection(aspect_ratio: InputTensor = 1.0,
                           znear: InputTensor = 0.0001,
                           zfar: InputTensor = 10,
                           fovy_degress: InputTensor = 60) -> t.Tensor:
  """Returns a 4x4 perspective projection matrix."""
  result = transformations.perspective_lh(fovy_degress * math.pi / 180,
                                          aspect_ratio, znear,
                                          zfar)
  # Invert the Y axis, since the origin in 2D in OpenGL is the top left corner.
  return t.matmul(transformations.scale((1, -1, 1)), result)


def get_ortho_matrix() -> t.Tensor:
  """Returns a 4x4 orthographic projection matrix."""
  return transformations.ortho_lh(-0.5, 0.5, 0.5, -0.5, -0.5, 0.5)


def get_default_camera_for_mesh(vertex_positions: InputTensor) -> t.Tensor:
  """Computes a default camera matrix, looking the object from above."""
  vertex_positions = util.to_tensor(vertex_positions, dtype=t.float32)
  assert vertex_positions.shape[-2:] == (3, 3)
  mesh_min = vertex_positions.reshape([-1, 3]).min(dim=0)[0]
  mesh_max = vertex_positions.reshape([-1, 3]).max(dim=0)[0]
  diagonal = (mesh_max - mesh_min).max()
  center = (mesh_min + mesh_max) / 2

  result = cameras_on_tetrahedron_vertices()[1]
  result = t.matmul(result, transformations.translate(-center))
  result = t.matmul(
      transformations.translate([0, 0, diagonal * 0.7]), result)

  projection_matrix = perspective_projection(
      1, zfar=diagonal * 3, znear=(diagonal + 10) / 1000)
  return t.matmul(projection_matrix, result)
