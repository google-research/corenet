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

"""CoreNet dataset routines."""

import dataclasses
import json
import math
from typing import Callable
from typing import Iterable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Text
from typing import Tuple
from typing import Union

import numpy as np
import pandas
import torch as t
import torch.utils.data
from dataclasses_jsonschema import JsonSchemaMixin

from corenet import file_system as fs
from corenet import misc_util
from corenet.data import scene

VOID_LABEL_NAME = "__void__"


@dataclasses.dataclass
class DatasetClass(JsonSchemaMixin):
  id: Text
  human_readable: Text


@dataclasses.dataclass
class DatasetConfig(JsonSchemaMixin):
  classes: List[DatasetClass]
  files: List[Text]


@dataclasses.dataclass
class DatasetElement(misc_util.TensorContainerMixin):
  """A single dataset element"""
  # The scene ID
  scene_id: str

  # The untransformed triangle vertices of all meshes
  # float32[num_total_tri, 3, 3]
  mesh_vertices: t.Tensor

  # The number of vertices in each mesh
  # int32[num_meshes]
  mesh_num_tri: t.Tensor

  # The world->view transform in the rendered images, float32[4, 4]
  view_transform: t.Tensor

  # The camera projection transform in the rendered images, float32[4, 4]
  camera_transform: t.Tensor

  # The object-to-world transform for each mesh, float32[num_meshes, 4, 4]
  o2w_transforms: t.Tensor

  # The mesh labels, int32[num_meshes].
  mesh_labels: t.Tensor

  # The rendered input image, uint8[3, height, width]
  input_image: t.Tensor


PipelineTransformation = Callable[
  [scene.Scene, DatasetElement], DatasetElement]
PipelineTransformations = Optional[List[PipelineTransformation]]


def to_dataset_element(
    ex: scene.Scene, file_name: str,
    class_to_int_mapping: Mapping[str, int],
    high_realism: bool
) -> DatasetElement:
  """Converts a scent to a dataset element."""
  input_image = misc_util.to_tensor(
      ex.pbrt_image if high_realism else ex.opengl_image,
      dtype=t.uint8).permute([2, 0, 1])

  mesh_labels = t.as_tensor(
      [class_to_int_mapping[v] for v in ex.mesh_labels],
      dtype=t.int32)
  mesh_num_tri = t.as_tensor([v.shape[0] for v in ex.mesh_vertices],
                             dtype=t.int32)
  mesh_vertices = t.cat(ex.mesh_vertices, dim=0)

  return DatasetElement(
      scene_id=fs.splitext(file_name)[0],
      mesh_vertices=mesh_vertices,
      mesh_num_tri=mesh_num_tri,
      view_transform=ex.view_transform,
      camera_transform=ex.camera_transform,
      o2w_transforms=ex.o2w_transforms,
      mesh_labels=mesh_labels,
      input_image=input_image
  )


def build_class_structures(
    dataset_config: DatasetConfig
) -> Tuple[Tuple[str, ...], Mapping[str, int]]:
  """Builds class structures from a dataset config

  Args:
    dataset_config: The dataset config,

  Returns:
    classes: The human readable class names, sorted, string[num_classes + 1].
      The first class is always "__void__"
    class_to_int_mapping: Mapping from a machine class ID, to index in
      "classes", the human readable class names returned by this function.
  """
  sorted_classes = sorted(
      dataset_config.classes, key=lambda v: v.human_readable)
  classes = tuple(
      [VOID_LABEL_NAME] + [v.human_readable for v in sorted_classes])
  class_to_int_mapping = {
      v.id: i + 1  # Class 0 is reserved for empty/void
      for i, v in enumerate(sorted_classes)
  }
  if (len(class_to_int_mapping) !=
      len(set(class_to_int_mapping.values()))):
    raise ValueError("Found duplicate class IDs")
  return classes, class_to_int_mapping


class CoReNetDatasetImpl(torch.utils.data.Dataset):
  """A CoreNet dataset, corresponding to a dataset on disk."""

  def __init__(self, dataset_path: Text, meshes_dir: Text,
               high_realism: bool = True,
               data_transforms: PipelineTransformations = None):
    """Creates a CoreNet dataset.

    Args:
      dataset_path: Path to a JSON file describing the dataset. Contains
        JSON serialized DatasetConfig.
      meshes_dir: Path to directory containing ShapeNet meshes.
      high_realism: Whether to use the high or the low realism image.
      data_transforms: List of transformations to apply to the data.
    """

    self.high_realism = high_realism
    self.data_transforms = data_transforms or []

    self.dataset_path = dataset_path
    self.meshes_dir = meshes_dir
    dataset_json = json.loads(fs.read_text(self.dataset_path))
    dataset_config = DatasetConfig.from_dict(dataset_json)

    self.root_directory = fs.dirname(self.dataset_path)
    self.classes, self.class_to_int_mapping = build_class_structures(
        dataset_config)

    # Avoids leaking memory in torch's DataLoaders, which is caused by to
    # copy on access in multiprocessing
    # https://github.com/pytorch/pytorch/issues/13246
    self.files = np.array(dataset_config.files)
    self.classes = np.array(self.classes)
    self.class_to_int_mapping = pandas.DataFrame(
        self.class_to_int_mapping, index=[0])

  def __getitem__(self, index: int) -> DatasetElement:
    file_name = self.files[index]
    inex = scene.load_from_npz(
        fs.join(self.root_directory, file_name),
        self.meshes_dir, load_extra_fields=False)
    dex = to_dataset_element(inex, file_name, self.class_to_int_mapping,
                             self.high_realism)
    # Correct for using a dataframe instead of a dict in the line above
    dex = dataclasses.replace(dex, mesh_labels=dex.mesh_labels.view(-1))
    for transf in self.data_transforms:
      dex = transf(inex, dex)
    return dex

  def __len__(self) -> int:
    return self.files.shape[0]


class CoReNetDataset(torch.utils.data.Dataset):
  """Virtual CoreNet dataset.

  Allows operations like slicing and concatenation, while preserving metadata.
  """

  def __init__(self, d: t.utils.data.Dataset,
               classes: Union[np.ndarray, Tuple[str, ...]],
               indices: Optional[t.Tensor] = None):
    self._dataset = d
    self.classes = np.array(classes)
    if indices is None:
      indices = t.arange(len(d), device="cpu")
    self.indices = indices

  def __add__(self, other: 'CoReNetDataset') -> 'CoReNetDataset':
    if other.classes != self.classes:
      raise ValueError("The classes of both datasets must match.")
    return concatenate([self, other])

  def __len__(self):
    return self.indices.shape[0]

  def __getitem__(
      self, index: Union[int, slice]
  ) -> Union[DatasetElement, 'CoReNetDataset']:
    if isinstance(index, slice):
      indices = self.indices[index]
      return CoReNetDataset(self._dataset, self.classes, indices)
    else:
      return self._dataset[int(self.indices[index])]

  def take_fraction(self, start: float, end: float) -> 'CoReNetDataset':
    assert 0 <= start <= end <= 1
    start_index = int(math.floor(start * len(self)))
    end_index = int(math.ceil(end * len(self)))
    return self[start_index: end_index]

  def shuffle(self, seed: int) -> 'CoReNetDataset':
    g = t.Generator()
    g.manual_seed(seed)
    indices = torch.randperm(self.indices.shape[0], generator=g, device="cpu")
    indices = self.indices[indices]
    return CoReNetDataset(self._dataset, self.classes, indices)


def concatenate(datasets: Iterable[CoReNetDataset]) -> CoReNetDataset:
  datasets = list(datasets)
  if len(datasets) == 1:
    return datasets[0]
  all_classes = np.array([v.classes for v in datasets])
  assert (all_classes[0:1] == all_classes).all()
  classes = all_classes[0]
  return CoReNetDataset(t.utils.data.ConcatDataset(datasets), classes)
