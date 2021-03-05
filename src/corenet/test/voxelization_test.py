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

# Lint as: python3
"""Unit tests for the simple renderer tensorflow operation."""

import unittest

import numpy as np
import numpy.testing as tt
import torch as t

from corenet.cc import fill_voxels
from corenet.geometry import transformations
from corenet.geometry import voxelization


def _create_cube_mesh(d: float):
  """Creates a cube slightly larger than the center voxel of a 3x3x3 grid."""
  m, x = [d, 3 - d]
  cube = [
      [[m, m, m], [m, x, m], [m, m, x]],
      [[m, x, x], [m, x, m], [m, m, x]],
      [[x, m, m], [x, x, m], [x, m, x]],
      [[x, x, x], [x, x, m], [x, m, x]],

      [[m, m, m], [m, m, x], [x, m, m]],
      [[x, m, x], [m, m, x], [x, m, m]],
      [[m, x, m], [m, x, x], [x, x, m]],
      [[x, x, x], [m, x, x], [x, x, m]],

      [[m, m, m], [m, x, m], [x, m, m]],
      [[x, x, m], [m, x, m], [x, m, m]],
      [[m, m, x], [m, x, x], [x, m, x]],
      [[x, x, x], [m, x, x], [x, m, x]],
  ]
  return t.tensor(cube, dtype=t.float32)


class VoxelizationTests(unittest.TestCase):

  def testVoxelizesSimpleExample(self):
    """Tests voxelization for a simple example with two triangles."""
    diagonal_quad = t.tensor([
        [  # Triangle 1
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        [  # Triangle 2
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 1],
        ]
    ], dtype=t.float32)
    expected_grid = t.tensor([
        [  # z = 0
            [1, 0, 0, 0],  # y = 0
            [1, 0, 0, 0],  # y = 1
            [1, 0, 0, 0],  # y = 2
            [1, 0, 0, 0],  # y = 3
        ],
        [  # z = 1
            [0, 1, 0, 0],  # y = 0
            [0, 1, 0, 0],  # y = 1
            [0, 1, 0, 0],  # y = 2
            [0, 1, 0, 0],  # y = 3
        ],
        [  # z = 2
            [0, 0, 1, 0],  # y = 0
            [0, 0, 1, 0],  # y = 1
            [0, 0, 1, 0],  # y = 2
            [0, 0, 1, 0],  # y = 3
        ],
        [  # z = 3
            [0, 0, 0, 1],  # y = 0
            [0, 0, 0, 1],  # y = 1
            [0, 0, 0, 1],  # y = 2
            [0, 0, 0, 1],  # y = 3
        ]
    ], dtype=t.float32)

    voxel_grid = voxelization.voxelize_mesh(diagonal_quad, [2], (4, 4, 4),
                                            transformations.scale([4, 4, 4]),
                                            image_resolution_multiplier=16)
    voxel_grid = voxel_grid.cuda()
    fill_voxels.fill_inside_voxels_gpu(voxel_grid, inplace=True)
    tt.assert_equal(voxel_grid.cpu().numpy(), expected_grid[np.newaxis])

  def testConservativeVoxelization(self):
    cube = _create_cube_mesh(99 / 100.0)
    grid = voxelization.voxelize_mesh(
        cube, [12], (3, 3, 3), transformations.scale([1, 1, 1]),
        image_resolution_multiplier=1)
    e = t.zeros([3, 3, 3])
    e[1, 1, [0, 2]] = e[1, [0, 2], 1] = e[[0, 2], 1, 1] = 1
    tt.assert_equal(e[None].numpy(), grid.numpy())

    grid = voxelization.voxelize_mesh(
        cube, [12], (3, 3, 3), transformations.scale([1, 1, 1]),
        image_resolution_multiplier=1, conservative_rasterization=True)
    e = t.ones([3, 3, 3])
    e[1, 1, 1] = 0
    tt.assert_equal(e[None].numpy(), grid.numpy())

  def testSubGridVoxelizationWorks(self):
    """Tests high-precision sub grid voxelization by shifting the geometry."""
    cube = _create_cube_mesh(99 / 100.0)
    grid = voxelization.voxelize_mesh(
        cube, [12], (3, 3, 3), transformations.scale([1, 1, 1]),
        sub_grid_sampling=True, image_resolution_multiplier=9,
        conservative_rasterization=True)
    grid = fill_voxels.fill_inside_voxels_gpu(grid.cuda(), inplace=False).cpu()
    e = t.zeros(1, 7, 7, 7)
    e[0, 2:5, 2:5, 2:5] = 1
    tt.assert_equal(e.numpy(), grid.numpy())
    grid = voxelization.get_sub_grid_centers(grid)
    e = t.zeros(1, 3, 3, 3)
    e[0, 1, 1, 1] = 1
    tt.assert_equal(e.numpy(), grid.numpy())

    cubes = t.cat([cube, cube - 0.5])
    transf = t.stack([transformations.translate([-0.5, 0, 0]),
                      transformations.translate([0.5, 1, 1])])
    grid = voxelization.voxelize_mesh(
        cubes, [12, 12], (3, 3, 3), transf,
        sub_grid_sampling=True, image_resolution_multiplier=9,
        conservative_rasterization=True)
    grid = fill_voxels.fill_inside_voxels_gpu(grid.cuda(), inplace=False).cpu()
    grid = voxelization.get_sub_grid_centers(grid)
    e1 = t.zeros(3, 3, 3)
    e1[1, 1, [0, 1]] = 1
    tt.assert_equal(e1.numpy(), grid[0].numpy())
    e2 = t.zeros(3, 3, 3)
    e2[1, [1, 2], 1] = e2[2, [1, 2], 1] = 1
    tt.assert_equal(e2.numpy(), grid[1].numpy())


class EmptyRegionFillTests(unittest.TestCase):

  def setUp(self):
    self.grid1 = t.tensor([
        [  # z = 0
            [1, 1, 1, 1],  # y = 0
            [1, 1, 1, 1],  # y = 1
            [1, 1, 1, 1],  # y = 2
            [1, 1, 1, 1],  # y = 3
        ],
        [  # z = 1
            [1, 1, 1, 1],  # y = 0
            [1, 0, 0, 1],  # y = 1
            [1, 0, 0, 1],  # y = 2
            [1, 1, 1, 1],  # y = 3
        ],
        [  # z = 2
            [1, 1, 1, 1],  # y = 0
            [1, 0, 0, 1],  # y = 1
            [1, 0, 0, 1],  # y = 2
            [1, 1, 1, 1],  # y = 3
        ],
        [  # z = 3
            [1, 1, 1, 1],  # y = 0
            [1, 1, 1, 1],  # y = 1
            [1, 1, 1, 1],  # y = 2
            [1, 1, 1, 1],  # y = 3
        ]
    ], dtype=t.float32)
    self.grid2 = t.tensor([
        [  # z = 0
            [1, 1, 1, 0],  # y = 0
            [1, 1, 1, 0],  # y = 1
            [1, 1, 1, 0],  # y = 2
            [0, 0, 0, 0],  # y = 3
        ],
        [  # z = 1
            [1, 1, 1, 0],  # y = 0
            [1, 0, 1, 0],  # y = 1
            [1, 1, 1, 0],  # y = 2
            [0, 0, 0, 0],  # y = 3
        ],
        [  # z = 2
            [1, 1, 1, 0],  # y = 0
            [1, 1, 1, 0],  # y = 1
            [1, 1, 1, 0],  # y = 2
            [0, 0, 0, 0],  # y = 3
        ],
        [  # z = 3
            [0, 0, 0, 0],  # y = 0
            [0, 0, 0, 0],  # y = 1
            [0, 0, 0, 0],  # y = 2
            [0, 0, 0, 0],  # y = 3
        ]
    ], dtype=t.float32)

  def testFillsCubeRegions(self):
    """Tests filling of cubes that are empty inside."""
    input_grid = t.stack([self.grid1, self.grid2], 0)
    # The OP is expected to fill inside regions with 2
    expected_grid1 = self.grid1.to(t.float32)
    expected_grid1[expected_grid1 == 0] = 1
    expected_grid2 = self.grid2.to(t.float32)
    expected_grid2[1, 1, 1] = 1
    expected_grid = t.stack([expected_grid1, expected_grid2], 0)

    voxel_grid = fill_voxels.fill_inside_voxels_gpu(
        input_grid.cuda(), inplace=False)
    tt.assert_equal(voxel_grid.cpu().numpy(), expected_grid.numpy())

  def testFillsCubeUint8Regions(self):
    """Tests filling of cubes that are empty inside."""
    input_grid = t.stack([self.grid1, self.grid2], 0).to(t.uint8)
    # The OP is expected to fill inside regions with 2
    expected_grid1 = self.grid1.to(t.uint8)
    expected_grid1[expected_grid1 == 0] = 1
    expected_grid2 = self.grid2.to(t.uint8)
    expected_grid2[1, 1, 1] = 1
    expected_grid = t.stack([expected_grid1, expected_grid2], 0)

    voxel_grid = fill_voxels.fill_inside_voxels_gpu(
        input_grid.cuda(), inplace=False)
    tt.assert_equal(voxel_grid.cpu().numpy(), expected_grid)

  def testGpuAndCpuReturnSame(self):
    """Tests filling of cubes that are empty inside."""
    input_grid = t.stack([self.grid1, self.grid2], 0)
    # The OP is expected to fill inside regions with 2
    expected_grid1 = self.grid1.to(t.float32)
    expected_grid1[expected_grid1 == 0] = 1
    expected_grid2 = self.grid2.to(t.float32)
    expected_grid2[1, 1, 1] = 1
    expected_grid = t.stack([expected_grid1, expected_grid2], 0)

    voxel_grid_gpu = fill_voxels.fill_inside_voxels_gpu(
        input_grid.cuda(), inplace=False)
    voxel_grid_cpu = fill_voxels.fill_inside_voxels_cpu(input_grid)
    tt.assert_equal(voxel_grid_gpu.cpu().numpy(), expected_grid.numpy())
    tt.assert_equal(voxel_grid_cpu.numpy(), expected_grid.numpy())


if __name__ == "__main__":
  unittest.main()
