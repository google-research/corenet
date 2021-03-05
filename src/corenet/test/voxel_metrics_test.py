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

import dataclasses
import unittest

import numpy.testing as tt
import torch as t

from corenet import voxel_metrics


class UtilTests(tt.TestCase):
  def testComputeConfusionMatrixProducesCorrectResult(self):
    gt = t.tensor([
        [[3, 2, 2, 4],
         [4, 3, 2, 2],
         [3, 1, 3, 0]],
        [[3, 0, 1, 3],
         [2, 3, 1, 1],
         [2, 3, 0, 4]]
    ], dtype=t.int32)
    pred = t.tensor([
        [[0, 2, 3, 1],
         [1, 1, 1, 3],
         [4, 0, 2, 3]],
        [[1, 0, 1, 4],
         [2, 4, 4, 0],
         [4, 2, 4, 2]]
    ], dtype=t.int32)
    confusion_matrix = voxel_metrics.confusion_matrix(pred, gt, 5)
    assert confusion_matrix.dtype == t.int32
    tt.assert_equal(
        confusion_matrix.numpy(),
        [[1, 0, 0, 1, 1],
         [2, 1, 0, 0, 1],
         [0, 1, 2, 2, 1],
         [1, 2, 2, 0, 3],
         [0, 2, 1, 0, 0]]
    )

  def testComputeTfpnProducesCorrectResult(self):
    confusion_matrix = t.tensor([
        [1, 0, 0, 1, 1],
        [2, 1, 0, 0, 1],
        [0, 1, 2, 2, 1],
        [1, 2, 2, 0, 3],
        [0, 2, 1, 0, 0]
    ], dtype=t.int32)

    tfpn = voxel_metrics.compute_tfpn(confusion_matrix)
    tt.assert_equal(tfpn.tp.numpy(), [1, 1, 2, 0, 0])
    tt.assert_equal(tfpn.tn.numpy(), [18, 15, 15, 13, 15])
    tt.assert_equal(tfpn.fp.numpy(), [3, 5, 3, 3, 6])
    tt.assert_equal(tfpn.fn.numpy(), [2, 3, 4, 8, 3])

  def testComputeMetricsProducesCorrectResult(self):
    tfpn = voxel_metrics.TfpnValues(
        tp=t.tensor([1, 1, 2, 0, 0], dtype=t.int32),
        tn=t.tensor([18, 15, 15, 13, 15], dtype=t.int32),
        fp=t.tensor([3, 5, 3, 3, 6], dtype=t.int32),
        fn=t.tensor([2, 3, 4, 8, 3], dtype=t.int32))

    mm = voxel_metrics.compute_voxel_metrics(tfpn)
    self.assertTrue(
        all([v.dtype == t.float64 for v in dataclasses.astuple(mm)]))
    tt.assert_allclose(mm.iou, [0.16666667, 0.11111111, 0.22222222, 0., 0.])
    tt.assert_allclose(mm.precision, [0.25, 0.16666667, 0.4, 0., 0.])
    tt.assert_allclose(mm.recall, [0.33333333, 0.25, 0.33333333, 0., 0.])


if __name__ == '__main__':
  unittest.main()
