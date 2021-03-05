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

"""Configuration classes that also map to JSON."""

import dataclasses
from typing import List
from typing import MutableMapping
from typing import Tuple
from typing import TypeVar

import enum
from dataclasses_jsonschema import JsonSchemaMixin

WITH_TEMPLATES_MARKER = "with_templates"


def templated_str_field() -> dataclasses.Field:
  """Enables template string substitution in configuration fields."""
  return dataclasses.field(metadata={WITH_TEMPLATES_MARKER: True})


class ShuffleType(enum.Enum):
  """How to shuffle a dataset."""

  # For training datasets, shuffles each epoch differently.
  # For eval datasets, shuffles differently for each different global step.
  PER_EPOCH = "PER_EPOCH"

  # Shuffles always in the same way, independent of the current epoch during
  # training and the global step during eval.
  ONCE = "ONCE"

  # Does not shuffle
  NEVER = "NEVER"


@dataclasses.dataclass
class Dataset(JsonSchemaMixin):
  """Dataset specification."""

  # Path to the dataset JSON definition
  dataset_path: str = templated_str_field()

  # Path to the root directory with ShapeNet meshes
  meshes_dir: str = templated_str_field()

  # Whether to use the high or low realism rendering
  high_realism: bool = True

  # How to shuffle the dataset. Shuffling is applied before before data_fraction
  shuffle: ShuffleType = ShuffleType.NEVER

  # Keeps only the first `data_fraction * len(dataset)` elements
  data_fraction: float = 1.0


class TaskType(enum.Enum):
  """The reconstruction task, determining the grid contents."""

  # The grid contains occupancy (FG/BG)
  FG_BG = "FG_BG"

  # The grid contains multinomial probability distribution over classes
  SEMANTIC = "SEMANTIC"


@dataclasses.dataclass
class DataLoaderConfig(JsonSchemaMixin):
  """Configures the data loader."""
  num_data_workers: int = 6
  batch_size: int = 4
  prefetch_factor: int = 2


@dataclasses.dataclass
class Resolution(JsonSchemaMixin):
  """Voxel grid resolution."""
  # Field order is important, must be D, H, W.
  depth: int
  height: int
  width: int


@dataclasses.dataclass
class VoxelizationConfig(JsonSchemaMixin):
  """Controls the point sampling of the scene volume.

  Sampling happens through voxelization, followed by flood filling of isolated
  empty inner regions. The center points of voxels that end up being full are
  considered full themselves, the rest are declared empty/void. This is an
  approximation of true point sampling. "Full" voxel centers with an adjacent
  empty voxel might in reality be "empty". And small holes in the mesh, with
  diameter comparable to the voxel side can be ignored/closed.

  We perform both voxelization and flood filling on the GPU. For the former, we
  use a method similar to
  https://developer.nvidia.com/content/basics-gpu-voxelization
  For the latter -- a CUDA implementation of connected components on for a
  regular grid graph.

  Higher resolutions reduce the approximation error but at the cost of cubic
  memory growth. We offer a mechanism to reduce the approximation error at a
  cost of constant increase of the memory (factor ~8), by voxelizing into a
  non-regular grid (see voxelization.voxelize_mesh for more detail).
  """

  # The reconstruction task type, which defines the contents of the grid
  task_type: TaskType

  # The grid resolution
  resolution: Resolution

  # Whether sub-grid sampling is enabled.
  sub_grid_sampling: bool = False

  # Whether to use conservative rasterization
  conservative_rasterization: bool = True

  # Controls the shape of the 2D render target used during voxelization, as a
  # function of the voxel grid shape. The shape will be square, with a side
  # equal to: `max(grid_depth, grid_height, grid_width) * this_multiplier`
  #
  # This multiplier must be odd when sub-grid sampling is enabled.
  # A value of `2*N+1` is equivalent to voxelizing into a grid with `2*N+1`
  # larger resolution on each side and then taking the sub-grid formed by the
  # centers of voxels (i*N+1, j*N+1, k*N+1).
  voxelization_image_resolution_multiplier: int = 5

  # Should be 1, except when evaluating the pre-trained model h7 from the paper,
  # which uses 2 (due to a bug). Affects the rounding error at object
  # boundaries. The difference between `1` and `2` is negligible (~0.003% of
  # all output voxels differ at 128^3). Nevertheless, to reproduce the
  # metrics reported in the paper exactly, use 2 for the pre-trained model h7.
  voxelization_projection_depth_multiplier: int = 1


@dataclasses.dataclass
class DataPipeline(JsonSchemaMixin):
  """Configures the data processing pipeline."""

  # These datasets get concatenated in order to form the final dataset
  datasets: List[Dataset]

  # How to shuffle the final (concatenated) dataset
  shuffle: ShuffleType

  # Data loader configuration
  data_loader: DataLoaderConfig

  # Volume sampling configuration
  voxelization_config: VoxelizationConfig


@dataclasses.dataclass
class EvalConfig(JsonSchemaMixin):
  """Configures an evaluation run"""

  # Name of the eval. Each eval will write results in a different sub-directory
  # with the name below.
  name: str

  # The eval data pipeline
  data: DataPipeline

  # How many qualitative results to write to disk for each eval run
  num_qualitative_results: int = 40

  # How many qualitative results to write to tensorboard for each eval run
  num_qualitative_results_in_tensor_board: int = 4


@dataclasses.dataclass
class StringTemplate(JsonSchemaMixin):
  """Defines a string substitution template argument with a default value. """
  key: str
  value: str = templated_str_field()


@dataclasses.dataclass
class TfModelEvalPipeline(JsonSchemaMixin):
  string_templates: List[StringTemplate]
  eval_config: EvalConfig
  frozen_graph_path: str = templated_str_field()
  output_path: str = templated_str_field()


@dataclasses.dataclass
class RecurrentEvalConfig(JsonSchemaMixin):
  """Recurrent evaluation during training.

  Attempts to evaluate after global step `start_step + K * interval` and before
  the next one. Note however that evaluation happens only between training
  iterations and the step size is the product of the train batch_size and the
  number of training GPUs. Thus, there is no guarantee that evals will run at
  the exact schedule above. However, any evals that should have ran inside a
  training iteration will be ran at the end of it.
  """

  # Starts evaluating after this global step. If this value is <0, the
  # evaluation will only run when manually evaluation is manually invoked.
  start_step: int

  # Evaluates every `interval` steps.
  interval: int

  # Whether to save a persistent checkpoint matching this eval
  persistent_checkpoint: bool

  # The actual eval config
  config: EvalConfig


@dataclasses.dataclass
class TrainConfig(JsonSchemaMixin):
  """Configures training."""

  # Data training pipeline config
  data: DataPipeline

  # Starting point, usually Resnet50 pre-trained on ImageNet.
  resnet50_imagenet_checkpoint: str = templated_str_field()

  # How often to save temporary checkpoints, measured in global steps.
  # Temporary checkpoints are regularly pruned and only the last few of them are
  # kept.
  # Note that there is no guarantee that a checkpoint will be saved at the exact
  # schedule of `K * checkpoint_interval`, since the size of a training
  # iteration is the product of the batch size and the number of GPUs.
  # If a checkpoint should have been saved inside an iteration, it will be saved
  # at the end of it.
  checkpoint_interval: int = 16000

  # How often to save persistent checkpoints. The note regarding iteration size
  # applies here as well.
  persistent_checkpoint_interval: int = 100000

  # How often to log training diagnostic information to tensorboard, measured
  # in global steps.
  tensorboard_log_interval: int = 1600

  # Optimizer settings
  initial_learning_rate: float = 0.01
  adam_epsilon: float = 1e-4

  # Whether to shift the point sampling positions randomly during training
  random_grid_offset: bool = True

  # Model parameters
  last_upscale_factor: int = 2
  latent_channels: int = 64
  skip_fraction: float = 0.75

  # Trains forever if max_steps is negative
  max_steps: int = -1


@dataclasses.dataclass
class TrainPipeline(JsonSchemaMixin):
  string_templates: List[StringTemplate]
  train: TrainConfig
  eval: List[RecurrentEvalConfig]
  output_path: str = templated_str_field()


@dataclasses.dataclass(frozen=True)
class DecoderConfig(JsonSchemaMixin):
  # (depth, height, width) of the output grid.
  resolution: Tuple[int, int, int]

  # Number of channels in the last (output) layer.
  num_output_channels: int

  # How much to upscale in the last layer of the grid.
  last_upscale_factor: int

  # Number of channels in the latent space.
  latent_channels: int

  # Fraction of additional channels from skip connections,
  # relative to the output of the previous layer. If 0, ray-traced skip
  # connections are disabled.
  skip_fraction: float


@dataclasses.dataclass(frozen=True)
class CoreNetConfig(JsonSchemaMixin):
  decoder: DecoderConfig


T = TypeVar('T')


def replace_templates(data: T, template_values: MutableMapping[str, str]) -> T:
  """Replaces string templates in all `templated_str_field`s."""
  if type(data) in {str, float, int, bool} or issubclass(type(data), enum.Enum):
    return data
  elif isinstance(data, list):
    return [replace_templates(v, template_values) for v in data]
  elif dataclasses.is_dataclass(data):
    fields = dataclasses.fields(data)  # type: List[dataclasses.Field]
    result = {}
    for f in fields:
      v = getattr(data, f.name)
      if WITH_TEMPLATES_MARKER in f.metadata:
        result[f.name] = v.format(**template_values)
      else:
        result[f.name] = replace_templates(v, template_values)

    result = type(data)(**result)
    # Processing a string template updates the dictionary
    if isinstance(result, StringTemplate):
      if result.key not in template_values:
        template_values[result.key] = result.value
    return result
  else:
    raise ValueError(f"Cannot handle data of type {type(data)}")


def parse_template_mapping(
    template_mapping: List[str]
) -> MutableMapping[str, str]:
  """Parses a string template map from <key>=<value> strings."""
  result = {}
  for mapping in template_mapping:
    key, value = mapping.split("=", 1)
    result[key] = value
  return result
