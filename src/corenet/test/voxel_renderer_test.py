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

"""Unit tests for the high level voxel rendering routines."""

import math
import unittest
from importlib import resources

import PIL.Image
import PIL.Image
import numpy as np
import torch as t

from corenet.geometry import transformations
from corenet.visualization import test_data
from corenet.visualization import voxel_renderer


class VoxelRendererTests(unittest.TestCase):

  def testRendersSimpleVoxelGrid(self):
    voxel_grid = t.as_tensor(
        [
            # z = 0
            [
                [1, 0, 1],  # y = 0
                [0, 0, 0],  # y = 1
                [1, 0, 1],  # y = 2
            ],
            # z = 1
            [
                [0, 0, 0],  # y = 0
                [0, 1, 0],  # y = 1
                [0, 0, 0],  # y = 2
            ],
            # z = 2
            [
                [0, 1, 0],  # y = 0
                [1, 1, 1],  # y = 1
                [0, 1, 0],  # y = 2
            ],
        ],
        dtype=t.int32)

    # Create a camera that looks at the voxel grid center from the side. The
    # 0.5 offset is required, since the voxel grid occupies the unit cube,
    # and its center is at  (0.5, 0.5, 0.5)
    look_at = transformations.look_at_rh((-1.2 + 0.5, -1.5 + 0.5, -0.5 + 0.5),
                                         (0.5, 0.5, 0.5), (0, 1, 0))
    perspective = transformations.perspective_rh(70 * math.pi / 180, 1, 0.1,
                                                 10.0)
    model_view_matrix = np.matmul(perspective, look_at)

    image = voxel_renderer.render_voxel_grid(
        voxel_grid,
        model_view_matrix,
        (256, 256),
        # Scale down the voxel grid to fit in the unit cube
        transformations.scale((1.0 / 3,) * 3),
        # Material 0 is transparent, 1 is red
        ((-1, 0, 0), (1.0, 0, 0)),
        # Place the light source at the camera
        light_position=(-1.2 + 0.5, -1.5 + 0.5, -1 + 0.5),
        ambient_light_color=(0.0, 0.0, 0.0),
    )
    image = image.numpy()

    PIL.Image.fromarray(image).save("/tmp/tt/vv.png")

    with resources.open_binary(test_data,
                               "expected_image_voxels.png") as in_file:
      pil_image = PIL.Image.open(in_file)
      expected_image = np.array(pil_image)[..., :3]

    self.assertEqual(image.dtype, np.uint8)
    self.assertEqual(tuple(image.shape), tuple(expected_image.shape))
    difference_l1 = np.abs(
        image.astype(np.int64) - expected_image.astype(np.int64)).sum()
    self.assertAlmostEqual(difference_l1, 0, 1024)


if __name__ == "__main__":
  unittest.main()
