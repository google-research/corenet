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

"""Routines for building training and evaluation pipelines."""

import dataclasses
import logging
from typing import Any
from typing import Dict

import io
import torch as t

from corenet import configuration as config_lib
from corenet import file_system as fs
from corenet.model import core_net

log = logging.getLogger(__name__)


@dataclasses.dataclass
class State:
  global_step: int
  model: core_net.CoreNet
  optimizer: t.optim.Optimizer
  extra_metadata: Any


@dataclasses.dataclass
class SavedState:
  global_step: int
  model_state: Dict[str, Any]
  model_config: Dict[str, Any]
  optimizer_state: Dict[str, Any]
  extra_metadata: Any


def create_initial_state(config: config_lib.TrainConfig,
                         num_classes: int) -> State:
  voxelization_config = config.data.voxelization_config
  num_channels = {
      config_lib.TaskType.SEMANTIC: num_classes,
      config_lib.TaskType.FG_BG: 2
  }[voxelization_config.task_type]
  model_config = config_lib.CoreNetConfig(
      decoder=config_lib.DecoderConfig(
          resolution=dataclasses.astuple(voxelization_config.resolution)[::-1],
          num_output_channels=num_channels,
          last_upscale_factor=config.last_upscale_factor,
          latent_channels=config.latent_channels,
          skip_fraction=config.skip_fraction))
  model = core_net.CoreNet(model_config)

  optimizer = t.optim.Adam(model.parameters(), lr=config.initial_learning_rate,
                           eps=config.adam_epsilon)
  initial_model_state = t.load(
      io.BytesIO(fs.read_bytes(config.resnet50_imagenet_checkpoint)))
  model.encoder.load_state_dict(initial_model_state)
  return State(global_step=0, model=model, optimizer=optimizer,
               extra_metadata=None)


def encode_state(state: State) -> bytes:
  state = SavedState(
      global_step=state.global_step, model_state=state.model.state_dict(),
      model_config=state.model.config.to_dict(),
      optimizer_state=state.optimizer.state_dict(),
      extra_metadata=state.extra_metadata)
  state = {k.name: getattr(state, k.name) for k in dataclasses.fields(state)}
  t.save(state, buf := io.BytesIO())
  return buf.getvalue()


def decode_state(raw_state: bytes, device: str) -> State:
  state = t.load(io.BytesIO(raw_state), map_location="cpu")
  state = SavedState(**state)
  model = core_net.CoreNet(
    config_lib.CoreNetConfig.from_dict(state.model_config))
  model.load_state_dict(state.model_state)
  model.to(device)

  optimizer = t.optim.Adam(model.parameters())
  optimizer.load_state_dict(state.optimizer_state)

  return State(global_step=state.global_step, model=model, optimizer=optimizer,
               extra_metadata=state.extra_metadata)
