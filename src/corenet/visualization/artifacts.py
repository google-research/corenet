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

"""Library for visualizing training/evaluation artifacts."""

import sys
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import skimage.measure
import torch as t
import torch.nn.functional as F
import torchvision.transforms.functional as vF

import corenet.misc_util as util
from corenet.geometry import transformations
from corenet.visualization import camera_util
from corenet.visualization import colors
from corenet.visualization import scene_renderer
from corenet.visualization import voxel_renderer


class VisualizationArtifact:
  """A visualization artifact.

  Artifacts allow 2D and 3D visualizations in a common coordinate frame.
  """

  def get_3d_box(self, transform: t.Tensor) -> Optional[t.Tensor]:
    """Computes the 3D box of an artifact under a given transformation.

    Args:
      transform: The transformation, float32[4, 4]
    Returns:
      The 3D box, float32[2, 3] if this is a 3D artifact. None otherwise.

    """
    raise NotImplementedError()

  def render(self, camera_matrix: t.Tensor,
             output_shape: Tuple[int, int]) -> t.Tensor:
    """Renders the artifact.


    Args:
      camera_matrix: The camera matrix, float32[4, 4]
      output_shape: The output image shape, (height, width)

    Returns:
      The rendered image, uint8[height, width, 3]

    """
    raise NotImplementedError()


class MultiMeshArtifact(VisualizationArtifact):
  """Artifact containing multiple meshes."""

  def __init__(self, vertices: t.Tensor, mesh_num_tri: t.Tensor,
               normals: Optional[t.Tensor] = None,
               mesh_colors: Optional[t.Tensor] = None):
    """Initializes the artifact.

    Accepts both tensors with and without a batch dimension.

    Args:
      vertices: The triangle vertices, float32[num_scene_tri, 3, 3]
      mesh_num_tri: The number of triangles in each mesh, int32[num_meshes]
      mesh_colors: The colors to use for the different meshes
        float32[num_meshes, 3]
    """
    vertices = util.to_tensor(vertices, t.float32)
    assert len(vertices.shape) == 3 and vertices.shape[1:] == (3, 3)
    device = vertices.device

    mesh_num_tri = util.to_tensor(mesh_num_tri, t.int32, device)
    assert (len(mesh_num_tri.shape) == 1 and
            mesh_num_tri.sum().item() == vertices.shape[0])
    if mesh_colors is None:
      mesh_colors = colors.DEFAULT_COLOR_PALETTE[1:mesh_num_tri.shape[0] + 1]
    mesh_colors = util.to_tensor(mesh_colors, t.float32, device)
    assert mesh_colors.shape == (mesh_num_tri.shape[0], 3)

    if normals is not None:
      normals = util.to_tensor(normals, t.float32, device)
      assert normals.shape == vertices.shape

    self.vertices = vertices
    self.normals = normals
    self.mesh_num_tri = mesh_num_tri
    self.mesh_colors = mesh_colors

  def get_3d_box(self, transform) -> Optional[t.Tensor]:
    transformed = transformations.transform_mesh(self.vertices, transform)
    ltf = transformed.reshape([-1, 3]).min(0)[0]
    rbb = transformed.reshape([-1, 3]).max(0)[0]
    return t.stack([ltf, rbb], 0)

  def render(self, camera_matrix: t.Tensor,
             output_shape: Tuple[int, int]) -> t.Tensor:
    material_ids = util.dynamic_tile(self.mesh_num_tri)
    return scene_renderer.render_scene(self.vertices, camera_matrix,
                                       output_shape, normals=self.normals,
                                       material_ids=material_ids,
                                       diffuse_coefficients=self.mesh_colors,
                                       cull_back_facing=False)


class VoxelGridArtifact(VisualizationArtifact):
  """A voxel grid artifact."""

  def __init__(self, voxel_grid: t.Tensor, voxel_to_world_transform: t.Tensor,
               palette: Optional[t.Tensor] = None, frame_label: int = -1):
    """Creates the artifact.

    Args:
      voxel_grid: The voxel grid, int32[depth, height, width]
        containing label of each voxel.
      voxel_to_world_transform: Matrix that converts from voxel to world space,
        float32[4, 4]
      palette: The label colors, float32[max_labels, 3]
      frame_label: Creates a cubic frame around the grid with label equal to
        frame_label, if frame_label is positive
    """
    voxel_grid = util.to_tensor(voxel_grid, t.int32)
    assert len(voxel_grid.shape) == 3
    device = voxel_grid.device

    voxel_to_world_transform = util.to_tensor(voxel_to_world_transform,
                                              t.float32, device)
    assert voxel_to_world_transform.shape == (4, 4)

    max_label = max(voxel_grid.max().item(), frame_label)
    if palette is None:
      palette = colors.DEFAULT_COLOR_PALETTE[1:max_label + 1]
    palette = util.to_tensor(palette, t.float32, device)
    assert palette.shape == (max_label, 3)
    void_color = palette.new_ones([1, 3], dtype=t.float32) * -1
    palette = t.cat([void_color, palette], 0)

    if frame_label > 0:
      voxel_grid = voxel_grid.clone()
      VoxelGridArtifact.draw_frame(voxel_grid, frame_label)

    self.voxel_grid = voxel_grid
    self.voxel_to_world_transform = voxel_to_world_transform
    self.palette = palette

  def get_3d_box(self, transform: t.Tensor) -> Optional[t.Tensor]:
    xyz = self.voxel_grid.nonzero(as_tuple=False).flip(-1)
    if xyz.shape[0] == 0:
      return transform.new_zeros([2, 3])
    combined = transformations.chain([transform, self.voxel_to_world_transform])
    xyz = transformations.transform_points(xyz + 0.5, combined)
    ltf = xyz.min(0)[0]
    rbb = xyz.max(0)[0]
    return t.stack([ltf, rbb], 0)

  def render(self, camera_matrix: t.Tensor,
             output_shape: Tuple[int, int]) -> t.Tensor:
    return voxel_renderer.render_voxel_grid(
        self.voxel_grid, camera_matrix, output_shape,
        voxel_to_view_matrix=self.voxel_to_world_transform,
        diffuse_coefficients=self.palette)

  @classmethod
  def draw_frame(cls, grid: t.Tensor, label: int) -> t.Tensor:
    """Creates a voxel grid with all corner voxels set to 1."""
    grid[:, 0, 0] = label
    grid[:, 0, -1] = label
    grid[:, -1, 0] = label
    grid[:, -1, -1] = label
    grid[0, :, 0] = label
    grid[0, :, -1] = label
    grid[-1, :, 0] = label
    grid[-1, :, -1] = label
    grid[0, 0, :] = label
    grid[0, -1, :] = label
    grid[-1, 0, :] = label
    grid[-1, -1, :] = label
    return grid


class MarchingCubesArtifact(VisualizationArtifact):
  """A marching cubes mesh."""

  def __init__(self,
               grid: t.Tensor,
               voxel_to_world: t.Tensor,
               palette: t.Tensor = None,
               filter_kernel: int = 1):
    """Initializes the artifact.

    Accepts both tensors with and without a batch dimension.

    Args:
      grid: float32[num_objects, depth, height, width].
      voxel_to_world: Matrix that converts from voxel to view space,
        float32[batch_size, 4, 4]
      palette: The colors to use for the different meshes. float32[batch_size,
        max_num_meshes, 3]
      filter_kernel: The size of the smoothing filter kernel to apply
    """
    grid = util.to_tensor(grid, dtype=t.float32)
    assert len(grid.shape) == 4

    voxel_to_world = util.to_tensor(voxel_to_world, t.float32, grid.device)
    assert voxel_to_world.shape == (4, 4)

    if filter_kernel > 1:
      k = filter_kernel
      grid = t.constant_pad_nd(grid, [(k - 1) // 2, k - 1 - (k - 1) // 2] * 3)
      kernel = grid.new_ones([1, 1, k, k, k], dtype=t.float32) / k ** 3
      grid = F.conv3d(grid[np.newaxis], kernel).squeeze(0)

    (vertices, normals,
     mesh_num_tri) = MarchingCubesArtifact.to_marching_cubes(grid[1:])

    vertices = transformations.transform_mesh(vertices, voxel_to_world, True)
    normals = transformations.transform_mesh(normals, voxel_to_world, False)
    if palette is not None:
      palette = palette[1:]
    self.mesh_artifact = MultiMeshArtifact(
        vertices=vertices, normals=normals, mesh_num_tri=mesh_num_tri,
        mesh_colors=palette)

  def get_3d_box(self, transform: t.Tensor) -> Optional[t.Tensor]:
    return self.mesh_artifact.get_3d_box(transform)

  def render(self, camera_matrix: t.Tensor,
             output_shape: Tuple[int, int]) -> t.Tensor:
    return self.mesh_artifact.render(camera_matrix, output_shape)

  @classmethod
  def to_marching_cubes(cls, voxel_grid: t.Tensor
                        ) -> Tuple[t.Tensor, t.Tensor, t.Tensor]:
    """Converts a voxel grid to a marching cubes mesh.

    Args:
      voxel_grid: The voxel grid, float32[num_objects, depth, height, width]

    Returns:
      vertices: The scene vertex positions, float32[num_triangles, 3, 3]
      normals: The scene vertex normals, float32[num_triangles, 3, 3]
      mesh_num_tri: The number of triangles in each mesh, int32[num_meshes]
    """

    voxel_grid = util.to_tensor(voxel_grid, dtype=t.float32)
    assert len(voxel_grid.shape) == 4

    triangles = []
    normals = []
    mesh_num_tri = []
    for grid in voxel_grid:
      grid = t.constant_pad_nd(grid, [1] * 6)
      if (grid > 0.5).sum() == 0:
        triangles.append(t.ones([1, 3, 3]))
        normals.append(t.ones([1, 3, 3]))
        mesh_num_tri.append(1)
        continue
      mc_result = skimage.measure.marching_cubes(
          grid.cpu().numpy(), level=0.5)
      mc_result = [t.as_tensor(v.copy()) for v in mc_result[:3]]
      vbuf, ibuf, nbuf = mc_result
      ibuf = ibuf.to(t.int64)
      assert ibuf.shape[0] > 0
      normals.append(nbuf[ibuf])
      triangles.append(vbuf[ibuf])
      mesh_num_tri.append(ibuf.shape[0])
    device = voxel_grid.device
    triangles = t.cat(triangles, dim=0).flip(-1).to(device)
    normals = t.cat(normals, dim=0).flip(-1).to(device)
    mesh_num_tri = util.to_tensor(mesh_num_tri, t.int32).to(device)
    return triangles, normals, mesh_num_tri


class ImageArtifact(VisualizationArtifact):
  """Image artifact."""

  def __init__(self, image: t.Tensor):
    """Initializes the artifact.

    Args:
      image: The images, uint8[3, height, width]
    """

    image = util.to_tensor(image, dtype=t.uint8)
    assert len(image.shape) == 3 and image.shape[0] == 3
    self.image = image

  def get_3d_box(self, transform: t.Tensor) -> Optional[t.Tensor]:
    return None

  def render(self, camera_matrix: t.Tensor,
             output_shape: Tuple[int, int]) -> t.Tensor:
    # Resize to output shape, preserving aspect ratio
    image = vF.to_pil_image(self.image.cpu())
    _, sh, sw = self.image.shape
    scale = min(output_shape[0] / sh, output_shape[1] / sw)
    th, tw = round(sh * scale), round(sw * scale)
    th, tw = min(th, output_shape[0]), min(tw, output_shape[1])
    result = vF.to_tensor(vF.resize(image, (th, tw)))  # type: t.Tensor
    result = (result * 255).clamp(0, 255).to(t.uint8).permute([1, 2, 0])
    pad_top, pad_left = (output_shape[0] - th) // 2, (output_shape[1] - tw) // 2
    result = t.constant_pad_nd(result, [pad_top, output_shape[0] - pad_top - th,
                                        pad_left,
                                        output_shape[0] - pad_left - tw, 0, 0])
    result = result.contiguous()
    return result


def compute_extra_views(
    artifacts: Iterable[VisualizationArtifact],
    aspect_ratio: float, world_to_view: t.Tensor
) -> List[t.Tensor]:
  """Computes extra viewpoints for the given artifacts group."""
  device = world_to_view.device
  ltf = t.tensor([sys.float_info.max] * 3, dtype=t.float32, device=device)
  rbb = t.tensor([-sys.float_info.max] * 3, dtype=t.float32, device=device)

  view_to_world = t.inverse(world_to_view)
  for artifact in artifacts:
    bbox = artifact.get_3d_box(view_to_world)
    if bbox is None:
      continue
    ltf = t.min(ltf, bbox[0])
    rbb = t.max(rbb, bbox[1])

  if (ltf > rbb).any():
    ltf = t.zeros_like(ltf)
    rbb = t.ones_like(rbb)

  center = (ltf + rbb) / 2
  diagonal = (rbb - ltf).max()

  tetrahedron_cameras = camera_util.cameras_on_tetrahedron_vertices()
  projection_matrix = camera_util.perspective_projection(aspect_ratio,
                                                         znear=0.01, zfar=10)
  result = [
      [
          projection_matrix,
          transformations.translate([0, 0, 0.3]),
          tetra_camera,
          transformations.scale([1 / diagonal] * 3),
          transformations.translate(-center),
          view_to_world
      ]
      for tetra_camera in tetrahedron_cameras
  ]

  return [
      transformations.chain([v.to(device) for v in transf_chain])
      for transf_chain in result
  ]


ArtifactOrGroup = Union[VisualizationArtifact, Iterable[VisualizationArtifact]]


def visualize_artifacts(
    artifacts: Iterable[ArtifactOrGroup], default_camera: t.Tensor,
    world_to_view: t.Tensor,
    image_size: Tuple[int, int] = (384, 384)
) -> List[t.Tensor]:
  """Visualizes the given artifacts.
  Args:
    artifacts: List of artifacts to visualize. All artifacts in a group will
      share the same extra 3D view points.
    default_camera: The default camera matrix for rendering the 3D artifacts.
    world_to_view: Artifacts are expected to be in view space. This specifies
      the world -> view transform.
    image_size: The output image size.

  Returns:
    List of rendered artifact images, one per camera. Each image contains all
    artifact renderings concatenated on the X axis.

  """
  camera_image_rows = [[] for x in range(5)]
  for group in artifacts:
    if isinstance(group, VisualizationArtifact):
      group = [group]
    cameras = [default_camera]
    cameras += compute_extra_views(group, image_size[1] / image_size[0],
                                   world_to_view)
    for i, camera in enumerate(cameras):
      for artifact in group:
        camera_image_rows[i].append(artifact.render(camera, image_size))
  camera_image_rows = [t.cat(v, 1) for v in camera_image_rows]
  return camera_image_rows
