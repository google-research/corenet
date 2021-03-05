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

"""Routines for handling batches during training/evaluation."""

import dataclasses
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import torch as t

from corenet import misc_util as util
from corenet.cc import fill_voxels
from corenet.data import dataset
from corenet.geometry import transformations
from corenet.geometry import voxelization


@dataclasses.dataclass(frozen=True)
class BatchedExample(util.TensorContainerMixin):
  """A batched training/evaluation example."""

  # The triangles of all scenes in the batch, float32[num_total_triangles, 3, 3]
  vertices: t.Tensor

  # The view transform applied to the scene, float32[batch_size, 4, 4]
  view_transform: t.Tensor

  # Camera transform for the contained geometry, float32[batch_size, 4, 4]
  camera_transform: t.Tensor

  # Number of triangles in each mesh, List[int32[num_meshes]].
  mesh_num_tri: List[t.Tensor]

  # The mesh labels, List[int32[num_meshes]].
  mesh_labels: List[t.Tensor]

  # The rendered scene, uint8[batch_size, 3, height, width]
  input_image: t.Tensor

  # The scene ID
  scene_id: List[str]

  # Where the voxels in the grid are sampled, relative to the origin of each
  # voxel, float32[batch_size, 3]. Range is [0, 1]^3.
  grid_sampling_offset: t.Tensor

  # The view->voxel transform used for the grid, float32[batch_size, 4, 4].
  v2x_transform: Optional[t.Tensor] = None

  # The voxel grid, int32[batch, depth, height, width].
  grid: Optional[t.Tensor] = None


def batch(examples: List[dataset.DatasetElement]) -> BatchedExample:
  """Batches a list of examples."""
  with t.no_grad():
    all_vertices = []
    batch_mesh_num_tri = []

    for ex in examples:
      w2v = ex.view_transform
      batch_mesh_num_tri.append(ex.mesh_num_tri)
      offset = 0
      for num_tri, o2w in zip(ex.mesh_num_tri, ex.o2w_transforms):
        mesh = ex.mesh_vertices[offset: offset + num_tri]
        offset += num_tri
        o2v = t.matmul(w2v, o2w)
        mesh = transformations.transform_mesh(mesh, o2v)
        all_vertices.append(mesh)
    all_vertices = t.cat(all_vertices, 0)
    return BatchedExample(
        vertices=all_vertices,
        view_transform=t.stack([e.view_transform for e in examples], 0),
        camera_transform=t.stack([e.camera_transform for e in examples], 0),
        mesh_num_tri=batch_mesh_num_tri,
        mesh_labels=[e.mesh_labels for e in examples],
        input_image=t.stack([e.input_image for e in examples], 0),
        scene_id=[e.scene_id for e in examples],
        grid_sampling_offset=all_vertices.new_ones(
            [len(batch_mesh_num_tri), 3]) * 0.5
    )


def voxel_content_mesh_index(batch_idx: int, mesh_idx: int) -> int:
  """Sets the voxel content to the mesh index."""
  _ = batch_idx
  return mesh_idx + 1


def voxel_content_1(batch_idx: int, mesh_idx: int) -> int:
  """Sets the voxel content to 1."""
  _ = batch_idx
  _ = mesh_idx
  return 1


class VoxelContentSemanticLabel:
  """Sets the voxel content to the mesh semantic class."""

  def __init__(self, semantic_labels: List[util.InputTensor]):
    self.semantic_labels = semantic_labels

  def __call__(self, batch_idx: int, mesh_idx: int, ) -> int:
    return self.semantic_labels[batch_idx][mesh_idx]


def voxelize(
    ex: BatchedExample,
    resolution: Tuple[int, int, int],
    voxel_content_fn: Callable[[int, int], int] = voxel_content_mesh_index,
    sub_grid_sampling: bool = False,
    conservative_rasterization: bool = False,
    image_resolution_multiplier=4,
    projection_depth_multiplier: int = 1,
    fill_inside: bool = True
) -> BatchedExample:
  """Voxelizes the batch geometry.

  Args:
    ex: The batch to voxelize.
    resolution: The grid resolution, tuple(depth, height, width)
    voxel_content_fn: A function (batch_index, mesh_index) => voxel_content that
      returns the value to be stored in a voxel, given a batch_index and a
      mesh_index.
    sub_grid_sampling: Allows approximate voxelization with much higher virtual
      resolution. Useful for testing whether points are inside an object or not.
    conservative_rasterization: Whether to enable conservative rasterization.
    image_resolution_multiplier: Determines the image resolution used to render
      the triangles as a function of the voxel grid resolution.
    projection_depth_multiplier: Should be 1. See the documentation of
      corenet.geometry.voxelization.voxelize_mesh
    fill_inside: Whether to fill the inside of the object

  Returns:
    The batch, with a replaced voxel grid.

  """
  with t.no_grad():
    d, h, w = resolution
    m = max(d, h, w)
    batch_size = ex.grid_sampling_offset.shape[0]

    # This is the world->voxel transform
    batch_v2x = (transformations.scale([m, m, m])
                 .expand([batch_size, 4, 4])
                 .to(ex.grid_sampling_offset.device))

    # Now compute the a shifted world->voxel transform to account for the fact
    # that we sample voxels at their centers in practice
    grid_shift = transformations.translate(ex.grid_sampling_offset - 0.5)
    shifted_w2x = t.matmul(grid_shift, batch_v2x)

    batch_num_meshes = [len(v) for v in ex.mesh_num_tri]
    mesh_v2x = []
    for num_meshes, w2x in zip(batch_num_meshes, shifted_w2x):
      mesh_v2x += [w2x] * num_meshes
    mesh_v2x = t.stack(mesh_v2x, 0)
    meshes_grid = voxelization.voxelize_mesh(
        triangles=ex.vertices, mesh_num_tri=t.cat(ex.mesh_num_tri, 0),
        resolution=resolution, view2voxel=mesh_v2x, cuda_device=None,
        sub_grid_sampling=sub_grid_sampling,
        image_resolution_multiplier=image_resolution_multiplier,
        conservative_rasterization=conservative_rasterization,
        projection_depth_multiplier=projection_depth_multiplier)
    # Allocate the output grid first, to reduce memory fragmentation
    output_grid = t.zeros([batch_size, d, h, w], dtype=t.int32, device="cuda")
    meshes_grid = meshes_grid.cuda()
    if fill_inside:
      fill_voxels.fill_inside_voxels_gpu(meshes_grid, inplace=True)

    if sub_grid_sampling:
      meshes_grid = voxelization.get_sub_grid_centers(meshes_grid)

    offset = 0
    for batch_idx, num_meshes in enumerate(batch_num_meshes):
      labels = [voxel_content_fn(batch_idx, m) for m in range(num_meshes)]
      labels = meshes_grid.new_tensor(labels, dtype=t.float32)
      labels = labels[:, None, None, None].expand(num_meshes, d, h, w)
      grid = labels * meshes_grid[offset:offset + num_meshes]
      offset += num_meshes
      grid = grid.max(dim=0)[0].to(t.int32)
      output_grid[batch_idx] = grid
    return dataclasses.replace(ex, v2x_transform=batch_v2x, grid=output_grid)
