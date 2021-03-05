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

"""Functions to compute transformation matrices."""
from typing import List

import torch as t
from torch.nn import functional as F

import corenet.misc_util as util
from corenet.misc_util import InputTensor


def scale(v: InputTensor) -> t.Tensor:
  """Computes a scale matrix.

  Args:
    v: The scale vector, float32[N].

  Returns:
    The scale matrix, float32[N+1, N+1]

  """
  v = util.to_tensor(v, dtype=t.float32)
  assert len(v.shape) == 1
  return t.diag(t.cat([v, v.new_ones([1])], dim=0))


def translate(v: InputTensor) -> t.Tensor:
  """Computes a translation matrix.

  Args:
    v: The translation vector, float32[B1, ..., BK, N].

  Returns:
    The translation matrix, float32[B1, ..., BK, N + 1, N + 1]

  """
  result = util.to_tensor(v, dtype=t.float32)
  assert len(result.shape) >= 1
  dimensions = result.shape[-1]
  result = result[..., None, :].transpose(-1, -2)
  result = t.constant_pad_nd(result, [dimensions, 0, 0, 1])
  id_matrix = t.diag(result.new_ones([dimensions + 1]))
  id_matrix = id_matrix.expand_as(result)
  result = result + id_matrix
  return result


def rotate(angle: InputTensor, axis: InputTensor) -> t.Tensor:
  """Computes a 3D rotation matrix.

  Args:
    angle: float32 scalar, specifying the angle in radians.
    axis: float32[3], specifying the rotatation axis

  Returns:
    The float32[4, 4] rotation matrix

  The formula used in this function is explained here:
  https://en.wikipedia.org/wiki/Rotation_matrix#Conversion_from_and_to_axis–angle
  """
  axis = util.to_tensor(axis, dtype=t.float32)
  angle = util.to_tensor(angle, dtype=t.float32)
  assert axis.shape == (3,)
  assert angle.shape == ()

  axis = F.normalize(axis, dim=-1)
  sin_axis = t.sin(angle) * axis
  cos_angle = t.cos(angle)
  cos1_axis = (1.0 - cos_angle) * axis
  _, axis_y, axis_z = t.unbind(axis, dim=-1)
  cos1_axis_x, cos1_axis_y, _ = t.unbind(cos1_axis, dim=-1)
  sin_axis_x, sin_axis_y, sin_axis_z = t.unbind(sin_axis, dim=-1)
  tmp = cos1_axis_x * axis_y
  m01 = tmp - sin_axis_z
  m10 = tmp + sin_axis_z
  tmp = cos1_axis_x * axis_z
  m02 = tmp + sin_axis_y
  m20 = tmp - sin_axis_y
  tmp = cos1_axis_y * axis_z
  m12 = tmp - sin_axis_x
  m21 = tmp + sin_axis_x
  zero = t.zeros_like(m01)
  one = t.ones_like(m01)
  diag = cos1_axis * axis + cos_angle
  diag_x, diag_y, diag_z = t.unbind(diag, dim=-1)

  matrix = t.stack((diag_x, m01, m02, zero, m10, diag_y, m12, zero, m20, m21,
                    diag_z, zero, zero, zero, zero, one),
                   dim=-1)
  output_shape = axis.shape[:-1] + (4, 4)
  result = matrix.reshape(output_shape)
  return result


def transform_points_homogeneous(points: InputTensor,
                                 matrix: InputTensor, w: float) -> t.Tensor:
  """Transforms a batch of 3D points with a batch of matrices.

  Args:
    points: The points to transform, float32[d1, ..., dn, num_points, 3]
    matrix: The transformation matrix, float32[d1, ..., dn, 4, 4]
    w: The W value to use. Should be 1 for affine points, 0 for vectors

  Returns:
    The transformed points in homogeneous space,
    float32[d1, ..., dn, num_points, 4]
  """
  points = util.to_tensor(points, dtype=t.float32)
  matrix = util.to_tensor(matrix, dtype=t.float32)
  assert points.shape[-1] == 3
  assert matrix.shape[-2:] == (4, 4)
  assert points.shape[:-2] == matrix.shape[:-2]

  batch_dims = points.shape[:-2]
  # Fold all batch dimensions into a single one
  points = points.reshape([-1] + list(points.shape[-2:]))
  matrix = matrix.reshape([-1] + list(matrix.shape[-2:]))

  points = t.constant_pad_nd(points, [0, 1], value=w)
  result = t.einsum("bnm,bvm->bvn", matrix, points)
  result = result.reshape(batch_dims + result.shape[-2:])

  return result


def transform_mesh(mesh: InputTensor, matrix: InputTensor,
                   vertices_are_points=True) -> t.Tensor:
  """Transforms a single 3D mesh.

  Args:
    mesh: The mesh's triangle vertices, float32[d1, ..., dn, num_tri, 3, 3]
    matrix: The transformation matrix, shape=[d1, ..., dn, 4, 4]
    vertices_are_points: Whether to interpret the vertices as points or vectors.

  Returns:
    The transformed mesh, same shape as input mesh

  """
  mesh = util.to_tensor(mesh, dtype=t.float32)
  matrix = util.to_tensor(matrix, dtype=t.float32)

  assert mesh.shape[-2:] == (3, 3)
  assert matrix.shape[-2:] == (4, 4)
  assert mesh.shape[:-3] == matrix.shape[:-2]

  original_shape = mesh.shape
  mesh = mesh.reshape([-1, mesh.shape[-3] * 3, 3])
  matrix = matrix.reshape([-1, 4, 4])
  w = 1 if vertices_are_points else 0
  mesh = transform_points_homogeneous(mesh, matrix, w=w)
  if vertices_are_points:
    mesh = mesh[..., :3] / mesh[..., 3:4]
  else:
    mesh = mesh[..., :3]

  return mesh.reshape(original_shape)


def transform_points(points: InputTensor,
                     matrix: InputTensor) -> t.Tensor:
  result = transform_points_homogeneous(points, matrix, w=1)
  result = result[..., :3] / result[..., 3:4]
  return result


def look_at_lh(eye: InputTensor, center: InputTensor,
               up: InputTensor) -> t.Tensor:
  """Computes a left-handed 4x4 look-at camera matrix."""
  eye = util.to_tensor(eye, dtype=t.float32)
  center = util.to_tensor(center, dtype=t.float32)
  up = util.to_tensor(up, dtype=t.float32)
  assert eye.shape == (3,)
  assert center.shape == (3,)
  assert up.shape == (3,)

  f = F.normalize(center - eye, dim=-1)
  s = F.normalize(t.cross(up, f), dim=-1)
  u = t.cross(f, s)

  return eye.new_tensor([
      [s[0], s[1], s[2], -t.dot(s, eye)],
      [u[0], u[1], u[2], -t.dot(u, eye)],
      [f[0], f[1], f[2], -t.dot(f, eye)],
      [0, 0, 0, 1],
  ], dtype=t.float32)


def look_at_rh(eye: InputTensor, center: InputTensor,
               up: InputTensor) -> t.Tensor:
  """Computes a right-handed 4x4 look-at camera matrix."""
  eye = util.to_tensor(eye, dtype=t.float32)
  center = util.to_tensor(center, dtype=t.float32)
  up = util.to_tensor(up, dtype=t.float32)
  assert eye.shape == (3,)
  assert center.shape == (3,)
  assert up.shape == (3,)

  f = F.normalize(center - eye, dim=-1)
  s = F.normalize(t.cross(f, up), dim=-1)
  u = t.cross(s, f)

  return eye.new_tensor([
      [s[0], s[1], s[2], -t.dot(s, eye)],
      [u[0], u[1], u[2], -t.dot(u, eye)],
      [-f[0], -f[1], -f[2], t.dot(f, eye)],
      [0, 0, 0, 1],
  ], dtype=t.float32)


def perspective_lh(fov_y: InputTensor, aspect: InputTensor, z_near: InputTensor,
                   z_far: InputTensor) -> t.Tensor:
  fov_y = util.to_tensor(fov_y, dtype=t.float32)
  aspect = util.to_tensor(aspect, dtype=t.float32)
  z_near = util.to_tensor(z_near, dtype=t.float32)
  z_far = util.to_tensor(z_far, dtype=t.float32)
  assert fov_y.shape == ()
  assert aspect.shape == ()
  assert z_near.shape == ()
  assert z_far.shape == ()

  tan_half_fov_y = t.tan(fov_y / 2)
  return fov_y.new_tensor([
      [1.0 / (aspect * tan_half_fov_y), 0, 0, 0],
      [0, 1.0 / tan_half_fov_y, 0, 0],
      [0, 0, (z_far + z_near) / (z_far - z_near),
       -(2 * z_far * z_near) / (z_far - z_near)],
      [0, 0, 1, 0],
  ], dtype=t.float32)


def perspective_rh(fov_y: InputTensor, aspect: InputTensor, z_near: InputTensor,
                   z_far: InputTensor) -> t.Tensor:
  fov_y = util.to_tensor(fov_y, dtype=t.float32)
  aspect = util.to_tensor(aspect, dtype=t.float32)
  z_near = util.to_tensor(z_near, dtype=t.float32)
  z_far = util.to_tensor(z_far, dtype=t.float32)
  assert fov_y.shape == ()
  assert aspect.shape == ()
  assert z_near.shape == ()
  assert z_far.shape == ()

  tan_half_fov_y = t.tan(fov_y / 2)
  return fov_y.new_tensor([
      [1.0 / (aspect * tan_half_fov_y), 0, 0, 0],
      [0, 1.0 / tan_half_fov_y, 0, 0],
      [0, 0, -(z_far + z_near) / (z_far - z_near),
       -(2 * z_far * z_near) / (z_far - z_near)],
      [0, 0, -1, 0],
  ], dtype=t.float32)


def ortho_lh(left: InputTensor, right: InputTensor, bottom: InputTensor,
             top: InputTensor, z_near: InputTensor, z_far: InputTensor
             ) -> t.Tensor:
  left = util.to_tensor(left, dtype=t.float32)
  right = util.to_tensor(right, dtype=t.float32)
  bottom = util.to_tensor(bottom, dtype=t.float32)
  top = util.to_tensor(top, dtype=t.float32)
  z_near = util.to_tensor(z_near, dtype=t.float32)
  z_far = util.to_tensor(z_far, dtype=t.float32)
  assert left.shape == ()
  assert right.shape == ()
  assert bottom.shape == ()
  assert top.shape == ()
  assert z_near.shape == ()
  assert z_far.shape == ()

  return left.new_tensor([
      [2 / (right - left), 0, 0, -(right + left) / (right - left)],
      [0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
      [0, 0, 2 / (z_far - z_near), - (z_far + z_near) / (z_far - z_near)],
      [0, 0, 0, 1]
  ], dtype=t.float32)


def chain(transforms: List[t.Tensor]) -> t.Tensor:
  assert transforms
  result = transforms[0]
  for transform in transforms[1:]:
    result = t.mm(result, transform)
  return result
