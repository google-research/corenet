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

import math
import unittest

import numpy.testing as tt
import torch as t

from corenet.geometry import transformations


class TransformationTests(tt.TestCase):

  def testScaleComputesCorrectMatrix(self):
    tt.assert_array_equal(
        transformations.scale((1, 2, 3)),
        t.tensor((
            (1, 0, 0, 0),
            (0, 2, 0, 0),
            (0, 0, 3, 0),
            (0, 0, 0, 1),
        ), dtype=t.float32))

  def testTranslateComputesCorrectMatrix(self):
    self.assertTrue(t.equal(
        transformations.translate((1, 2, 3)),
        t.tensor((
            (1, 0, 0, 1),
            (0, 1, 0, 2),
            (0, 0, 1, 3),
            (0, 0, 0, 1),
        ), dtype=t.float32)))
    self.assertTrue(t.equal(
        transformations.translate([[[1, 2, 3], [4, 5, 6]]]),
        t.tensor([[[
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1],
        ], [
            [1, 0, 0, 4],
            [0, 1, 0, 5],
            [0, 0, 1, 6],
            [0, 0, 0, 1],
        ]]], dtype=t.float32)))

  def testRotateComputesCorrectMatrix(self):
    tt.assert_allclose(
        transformations.rotate(math.pi / 2, (0, 0, 1)),
        t.tensor((
            (0, -1, 0, 0),
            (1, 0, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        ), dtype=t.float32),
        rtol=1e-5, atol=1e-5)

  def testTransformPointsHomogeneousCorrectlyTransformsPoints(self):
    transform_1 = (
        (1, 0, 0, 0),
        (0, 2, 0, 0),
        (0, 0, 3, 0),
        (0, 0, 0, 1),
    )
    transform_2 = (
        (1, 0, 0, 1),
        (0, 1, 0, 2),
        (0, 0, 1, 3),
        (0, 0, 0, 1),
    )
    points_1 = ((12, 34, 56), (34, 32, 30), (11, 11, 18), (5, 6, 7))
    points_2 = ((1, 2, 3), (4, 5, 6), (6, 5, 4), (3, 2, 1))

    expected_result = t.tensor((
        ((12, 68, 168), (34, 64, 90), (11, 22, 54), (5, 12, 21)),
        ((2, 4, 6), (5, 7, 9), (7, 7, 7), (4, 4, 4)),
    ), dtype=t.float32)

    transformed_points = transformations.transform_points_homogeneous(
        (points_1, points_2), (transform_1, transform_2), w=1)
    transformed_points = (
        transformed_points[..., :3] / transformed_points[..., 3:4])
    self.assertTrue(t.equal(transformed_points, expected_result))

  def testTransformMeshCorrectlyTransformsMeshes(self):
    transform = (
        (1, 0, 0, 0),
        (0, 2, 0, 0),
        (0, 0, 3, 0),
        (0, 0, 0, 1),
    )
    mesh = (
        ((12, 34, 56), (34, 32, 30), (11, 11, 18)),
        ((1, 2, 3), (4, 5, 6), (6, 5, 4)),
    )

    expected_result = t.tensor((
        ((12, 68, 168), (34, 64, 90), (11, 22, 54)),
        ((1, 4, 9), (4, 10, 18), (6, 10, 12)),
    ), dtype=t.float32)
    transformed_mesh = transformations.transform_mesh(mesh, transform)
    self.assertTrue(t.equal(transformed_mesh, expected_result))


if __name__ == '__main__':
  unittest.main()
