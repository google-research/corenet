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

"""Unit tests for tf_transformations.py."""

import unittest

import numpy.testing as tt
from corenet.visualization import camera_util


class UtilTests(tt.TestCase):

  def testCamerasOnTetrahedronVerticesComputesCorrectMatrix(self):
    tt.assert_allclose(
        camera_util.cameras_on_tetrahedron_vertices(),
        ((
             (0., -0., 1., -0.),
             (-0.33333, 0.94281, 0., -0.),
             (-0.94281, -0.33333, 0., 1.),
             (0., 0., 0., 1.),
         ), (
             (-0.86603, 0., -0.5, -0.),
             (0.16667, 0.94281, -0.28868, 0.),
             (0.4714, -0.33333, -0.8165, 1.),
             (0., 0., 0., 1.),
         ), (
             (0.86603, 0., -0.5, -0.),
             (0.16667, 0.94281, 0.28868, 0.),
             (0.4714, -0.33333, 0.8165, 1.),
             (0., 0., 0., 1.),
         ), (
             (-0., 0., 1., -0.),
             (1., -0., 0., -0.),
             (0., 1., 0., 1.),
             (0., 0., 0., 1.),
         )), atol=1e-4)

  def testFrontalCameraComputesCorrectMatrix(self):
    tt.assert_allclose(
        camera_util.frontal_camera(1.5), (
            (-1., 0., 0., -0.),
            (0., 1., 0., -0.),
            (0., 0., -1., 1.5),
            (0., 0., 0., 1.),
        ))

  def testPerspectiveProjectionComputesCorrectMatrix(self):
    tt.assert_allclose(camera_util.perspective_projection(), (
        (1.73205, 0., 0., 0.),
        (0., -1.73205, 0., 0.),
        (0., 0., 1.00002, -0.0002),
        (0., 0., 1., 0.),
    ), rtol=1e-5, atol=1e-5)

  def testGetOrthoMatrixComputesCorrectMatrix(self):
    tt.assert_allclose(camera_util.get_ortho_matrix(), (
        (2., 0., 0., -0.),
        (0., -2., 0., 0.),
        (0., 0., 2., 0.),
        (0., 0., 0., 1.),
    ))

  def testGetDefaultCameraForMeshComputesCorrectMatrix(self):
    mesh = (
        ((12, 34, 56), (34, 32, 30), (11, 11, 18)),
        ((1, 2, 3), (4, 5, 6), (6, 5, 4)),
    )
    tt.assert_allclose(
        camera_util.get_default_camera_for_mesh(mesh), (
            (-1.4999999, 0., -0.86602527, 51.79774),
            (-0.2886751, -1.632993, 0.49999994, 19.695688),
            (0.47177827, -0.33359763, -0.817144, 59.858547),
            (0.47140452, -0.33333334, -0.8164966, 59.937073),
        ), rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
  unittest.main()
