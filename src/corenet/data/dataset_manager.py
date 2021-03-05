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

import logging
from typing import Any
from typing import Dict

from corenet import configuration
from corenet import file_system as fs
from corenet.data import dataset as dataset_lib

log = logging.getLogger(__name__)


def _dataset_path(d: configuration.Dataset):
  return fs.normpath(fs.abspath(d.dataset_path))


def _dataset_key(d: configuration.Dataset):
  return _dataset_path(d), d.meshes_dir, d.high_realism


class DatasetManager:
  dataset_cache: Dict[Any, dataset_lib.CoReNetDatasetImpl] = {}

  def __init__(self, data_pipeline: configuration.DataPipeline,
               global_seed=0x5678):

    self.data_pipeline = data_pipeline
    self.global_seed = global_seed
    for d in data_pipeline.datasets:
      key = _dataset_key(d)
      ds_path = _dataset_path(d)
      if key not in self.dataset_cache:
        log.info(f"Reading dataset {ds_path}...")
        dataset = dataset_lib.CoReNetDatasetImpl(
            dataset_path=ds_path,
            meshes_dir=d.meshes_dir,
            high_realism=d.high_realism)
        self.dataset_cache[key] = dataset

    self.epoch_len = 0
    for d in self.data_pipeline.datasets:
      dataset = self.dataset_cache[_dataset_key(d)]
      dataset = dataset_lib.CoReNetDataset(dataset, dataset.classes)
      dataset = dataset.take_fraction(0, d.data_fraction)
      self.classes = dataset.classes
      self.epoch_len += len(dataset)

  def create_dataset(self, local_seed: int = 0x1234):
    result = []
    local_seed = (local_seed * 19 + 317)
    for d in self.data_pipeline.datasets:
      dataset = self.dataset_cache[_dataset_key(d)]
      dataset = dataset_lib.CoReNetDataset(dataset, dataset.classes)
      if d.shuffle == configuration.ShuffleType.ONCE:
        dataset = dataset.shuffle(self.global_seed)
      elif d.shuffle == configuration.ShuffleType.PER_EPOCH:
        dataset = dataset.shuffle(local_seed)
      dataset = dataset.take_fraction(0, d.data_fraction)
      result.append(dataset)

    result = dataset_lib.concatenate(result)
    if self.data_pipeline.shuffle == configuration.ShuffleType.ONCE:
      result = result.shuffle(self.global_seed)
    elif self.data_pipeline.shuffle == configuration.ShuffleType.PER_EPOCH:
      result = result.shuffle(local_seed)
    return result

  def create_dataset_from_start_step(self, start_step: int):
    epoch = start_step // self.epoch_len
    start_step_in_epoch = start_step % self.epoch_len
    return self.create_dataset(local_seed=epoch)[start_step_in_epoch:]
