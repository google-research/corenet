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

"""Routines for reading synthetic 3D scenes."""

import dataclasses as d
from typing import Any
from typing import List
from typing import Optional
from typing import Text

import PIL.Image
import io
import numpy as np
import torch as t

from corenet import file_system as fs
from corenet import misc_util as util


@d.dataclass(frozen=True)
class Scene(util.TensorContainerMixin):
  """A rendered synthetic scene."""

  # The untransformed triangle vertices of each mesh,
  # List[float32[num_triangles, 3, 3]]
  mesh_vertices: List[t.Tensor]

  # The view transform applied to the scene, float32[4, 4]
  view_transform: t.Tensor

  # The object-to-world transforms, float32[num_meshes, 4, 4]
  o2w_transforms: t.Tensor

  # Camera transform for the contained geometry, float32[4, 4]
  camera_transform: t.Tensor

  # The mesh labels, string[num_meshes].
  mesh_labels: List[Text]

  # Visible fraction of each mesh in the image, float32[num_meshes].
  mesh_visible_fractions: t.Tensor

  # An eye-lit image rendered with OpenGL, uint8[height, width, 3]
  opengl_image: t.Tensor

  # Scene rendered with global illumination (using PBRT),
  # uint8[height, width, 3]
  pbrt_image: t.Tensor

  # The untransformed mesh normals, List[float32[num_triangles, 3, 3]]
  normals: List[t.Tensor] = d.field(default_factory=lambda: [])

  # The mesh texcoords, List[float32[num_triangles, 3, 2]]
  texcoords: List[t.Tensor] = d.field(default_factory=lambda: [])

  # The mesh material ids, List[int32[num_triangles]]
  material_ids: List[t.Tensor] = d.field(default_factory=lambda: [])

  # The mesh diffuse material colors, float32[num_materials, 3]
  diffuse_colors: List[t.Tensor] = d.field(default_factory=lambda: [])

  # The diffuse texture PNGs, List[string[num_materials]]. An empty string here
  # corresponds to a material without a texture
  diffuse_texture_pngs: List[List[bytes]] = d.field(default_factory=lambda: [])


def _load_image(i):
  return util.to_tensor(np.array(PIL.Image.open(io.BytesIO(i))), t.uint8)


class NpzReader:
  def __init__(self, path: str):
    # noinspection PyTypeChecker
    self.npz = np.load(io.BytesIO(fs.read_bytes(path)), allow_pickle=True)

  def tensor(self, item: str, dtype: Optional[t.dtype] = None) -> t.Tensor:
    result = self.npz[item]  # type: np.ndarray
    if dtype:
      return util.to_tensor(result, dtype)
    else:
      return t.as_tensor(result)

  def list(self, item: str) -> List[Any]:
    result = self.npz[item]  # type: np.ndarray
    assert len(result.shape) == 1
    return list(result)

  def scalar(self, item: str) -> Any:
    result = self.npz[item]  # type: np.ndarray
    assert len(result.shape) == 0
    return result


def load_from_npz(path: Text, meshes_dir: Text,
                  load_extra_fields=False) -> Scene:
  """Loads an input example.

  Args:
    path: Path to NPZ with scene.
    meshes_dir: Path containing ShapeNet meshes.
    load_extra_fields: Whether to load extra fields that are not required for
      running the pipeline (e.g. texture coordinates)

  Returns:
    The loaded input example.

  """
  scene_npz = NpzReader(path)
  mesh_paths = [
      fs.join(meshes_dir, *v) + ".npz"
      for v in
      zip(scene_npz.list("mesh_labels"), scene_npz.list("mesh_filenames"))]

  result = Scene(
      mesh_vertices=[],
      view_transform=scene_npz.tensor("view_transform", t.float32),
      o2w_transforms=scene_npz.tensor("mesh_object_to_world_transforms",
                                      t.float32),
      camera_transform=scene_npz.tensor("camera_transform", t.float32),
      mesh_labels=[v for v in scene_npz.list("mesh_labels")],
      opengl_image=_load_image(scene_npz.scalar("opengl_image")),
      pbrt_image=_load_image(scene_npz.scalar("pbrt_image")),
      mesh_visible_fractions=scene_npz.tensor("mesh_visible_fractions",
                                              t.float32),
  )

  for mesh_path in mesh_paths:
    # noinspection PyTypeChecker
    mesh_npz = NpzReader(mesh_path)
    result.mesh_vertices.append(mesh_npz.tensor("vertices", t.float32))

    if load_extra_fields:
      result.normals.append(mesh_npz.tensor("normals", t.float32))
      result.material_ids.append(mesh_npz.tensor("material_ids", t.int32))
      result.texcoords.append(mesh_npz.tensor("texcoords", t.float32))
      result.diffuse_colors.append(mesh_npz.tensor("diffuse_colors", t.float32))
      result.diffuse_texture_pngs.append(
        mesh_npz.scalar("diffuse_texture_pngs"))
  return result
