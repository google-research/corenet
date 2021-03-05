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

import unittest

import numpy.testing as tt
import torch as t

from corenet.model import losses


class LossesTests(unittest.TestCase):

  def setUp(self):
    self.logits = t.as_tensor([
        [[[[0.8278376, 0.44923675, 0.9302666, 0.6919297],
           [0.38287663, 0.37834585, 0.051413298, 0.7789054]],
          [[0.71893823, 0.94472325, 0.35577738, 0.0018994808],
           [0.41523135, 0.7561617, 0.0044674873, 0.38063014]],
          [[0.3408773, 0.22092032, 0.0767951, 0.17644858],
           [0.3457942, 0.27810383, 0.74627364, 0.43618906]]],
         [[[0.70214736, 0.54277015, 0.4549327, 0.79017854],
           [0.4176488, 0.22357666, 0.43264854, 0.29656994]],
          [[0.15031266, 0.8952414, 0.011986375, 0.26919663],
           [0.084516525, 0.043944597, 0.6917249, 0.5230026]],
          [[0.42145348, 0.28770554, 0.50909555, 0.48172605],
           [0.97358274, 0.8910786, 0.5946312, 0.51896834]]]],
        [[[[0.9724301, 0.41606557, 0.1918621, 0.1327486],
           [0.6457069, 0.76746213, 0.022811055, 0.8097471]],
          [[0.44591904, 0.51651776, 0.89206624, 0.98763657],
           [0.75536454, 0.20767283, 0.01293385, 0.57412446]],
          [[0.551981, 0.2299962, 0.40206707, 0.7424828],
           [0.16304898, 0.26685357, 0.10787654, 0.48786318]]],
         [[[0.97532773, 0.52998006, 0.5693196, 0.28751576],
           [0.22973418, 0.5575429, 0.5877949, 0.461349]],
          [[0.320073, 0.69799054, 0.41638315, 0.13438594],
           [0.015848756, 0.45914185, 0.40993977, 0.031940937]],
          [[0.13979805, 0.24647367, 0.8555057, 0.40757453],
           [0.70918477, 0.9841, 0.93651617, 0.42834997]]]]
    ], dtype=t.float32)
    self.logits = self.logits.permute([0, 4, 1, 2, 3])

    self.gt = t.as_tensor([
        [[[2, 0], [2, 2], [0, 3]],
         [[0, 2], [0, 3], [3, 2]]],
        [[[2, 2], [3, 1], [2, 2]],
         [[1, 2], [1, 2], [1, 3]]]
    ], dtype=t.int64)

    self.weights = t.as_tensor([
        [[[0.19875002, 0.77583194], [0.5079423, 0.10823226],
          [0.84881544, 0.38121593]],
         [[0.32796824, 0.6824727], [0.9398581, 0.45499086],
          [0.4005183, 0.025895357]]],
        [[[0.77079856, 0.5860559], [0.15548718, 0.40526056],
          [0.21678174, 0.81268084]],
         [[0.77574897, 0.27733755], [0.1688559, 0.69102776],
          [0.5144435, 0.42727184]]]
    ], dtype=t.float32)

  def testIoUAgnosticLoss(self):
    tt.assert_allclose(losses.iou_agnostic(self.gt, self.logits), 0.8060565,
                       rtol=1e-5, atol=1e-6)
    tt.assert_allclose(losses.iou_agnostic(self.gt, self.logits, self.weights),
                       0.8174121, rtol=1e-5, atol=1e-6)

  def testIoUFgBgLoss(self):
    tt.assert_allclose(losses.iou_fgbg(self.gt, self.logits), 0.3579613,
                       rtol=1e-5, atol=1e-6)
    tt.assert_allclose(losses.iou_fgbg(self.gt, self.logits, self.weights),
                       0.4265449, rtol=1e-5, atol=1e-6)

  def testXEntLoss(self):
    tt.assert_allclose(losses.xent(self.gt, self.logits), 1.4547757, rtol=1e-5,
                       atol=1e-6)
    tt.assert_allclose(losses.xent(self.gt, self.logits, self.weights),
                       0.7043564, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
  unittest.main()
