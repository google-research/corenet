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

import dataclasses
import json
import os.path
import pathlib
from typing import Any
from typing import List
from typing import NamedTuple

import json5

import corenet.configuration as c
import corenet.data.dataset as dataset_lib


class AllDataSets(NamedTuple):
  single_train: Any
  single_val: Any
  single_test: Any
  pairs_train: Any
  pairs_val: Any
  pairs_test: Any
  triplets_train: Any
  triplets_val: Any
  triplets_test: Any


def lo_realism(all_ds: AllDataSets) -> AllDataSets:
  return AllDataSets(
      *[dataclasses.replace(ds, high_realism=False) for ds in all_ds])


def shuffle_per_epoch(ds: c.Dataset) -> c.Dataset:
  return dataclasses.replace(ds, shuffle=c.ShuffleType.PER_EPOCH)


def create_data_loader():
  """Default data loader, used for training and evaluation of all models."""
  return c.DataLoaderConfig(num_data_workers=6, batch_size=4)


def create_evals(all_ds: AllDataSets, num_obj: int, vox: c.VoxelizationConfig):
  """Creates recurrent evaluation configurations for a model.

  Args:
    all_ds: All datasets
    num_obj: The number of objects in the scene
    vox: Voxelization configuration

  Returns:
    Array of recurrent evaluation configurations.

  """
  ds_name = {1: "single", 2: "pairs", 3: "triplets"}[num_obj]
  ds_test = getattr(all_ds, f"{ds_name}_test")  # type: c.Dataset
  ds_val = getattr(all_ds, f"{ds_name}_val")  # type: c.Dataset
  assert ds_test.shuffle == c.ShuffleType.ONCE
  # 1% of test data
  ds_test_short = dataclasses.replace(ds_test, data_fraction=1e-2)
  # 10% of test data
  ds_test_medium = dataclasses.replace(ds_test, data_fraction=1e-1)
  # 1% of val data, which is also used for training
  ds_short_train = dataclasses.replace(ds_val, data_fraction=1e-2,
                                       shuffle=c.ShuffleType.ONCE)
  return [
      # Frequently run evaluation on a small fraction of the train data. Data
      # is shuffled in a stable way (independent of the eval run). This
      # guarantees that the exact same examples appear in all eval runs, which
      # allows tracking their progress both quantitatively and qualitatively.
      c.RecurrentEvalConfig(
          start_step=40000, interval=40000, persistent_checkpoint=False,
          config=c.EvalConfig(
              name="short_stable_train_eval",
              num_qualitative_results=40,
              num_qualitative_results_in_tensor_board=4,
              data=c.DataPipeline(
                  datasets=[ds_short_train], data_loader=create_data_loader(),
                  voxelization_config=vox, shuffle=c.ShuffleType.ONCE))),

      # Frequently run evaluation on a small fraction of all test data. Data
      # is shuffled in a stable way (independent of the eval run). This
      # guarantees that the exact same examples appear in all eval runs, which
      # allows tracking their progress both quantitatively and qualitatively.
      c.RecurrentEvalConfig(
          start_step=40000, interval=40000, persistent_checkpoint=False,
          config=c.EvalConfig(
              name="short_stable_eval",
              num_qualitative_results=40,
              num_qualitative_results_in_tensor_board=4,
              data=c.DataPipeline(
                  datasets=[ds_test_short], data_loader=create_data_loader(),
                  voxelization_config=vox, shuffle=c.ShuffleType.ONCE))),

      # Less frequently run evaluation on a larger fraction of the test data.
      # Data is shuffled differently in the different eval runs, which means
      # that each eval run will see a different set of examples. This allows
      # to judge current model performance in an unbiased way.
      c.RecurrentEvalConfig(
          start_step=140000, interval=140000, persistent_checkpoint=False,
          config=c.EvalConfig(
              name="medium_eval",
              num_qualitative_results=100,
              num_qualitative_results_in_tensor_board=4,
              data=c.DataPipeline(
                  datasets=[shuffle_per_epoch(ds_test_medium)],
                  data_loader=create_data_loader(), voxelization_config=vox,
                  shuffle=c.ShuffleType.PER_EPOCH))),

      # Full evaluation run, which always sees all test data. Data is shuffled
      # in a stable way, which guarantees that the exact same qualitative
      # examples are saved in all eval runs. This allows tracking their progress
      # both qualitatively.
      c.RecurrentEvalConfig(
          start_step=500000, interval=500000, persistent_checkpoint=True,
          config=c.EvalConfig(
              name="full_eval",
              num_qualitative_results=500,
              num_qualitative_results_in_tensor_board=0,
              data=c.DataPipeline(
                  datasets=[ds_test], data_loader=create_data_loader(),
                  voxelization_config=vox, shuffle=c.ShuffleType.ONCE)))
  ]


schema_paths = {
    c.TrainPipeline: "../schemas/train_config.json",
    c.TfModelEvalPipeline: "../schemas/tf_model_eval_config.json"
}


def dumps(p: c.JsonSchemaMixin):
  d = p.to_dict()
  d["$schema"] = schema_paths[type(p)]
  result = json5.dumps(d, indent=2)
  result = (
      f"//Generated automatically, by {os.path.basename(__file__)}\n{result}")
  return result


def generate_default_datasets() -> AllDataSets:
  """Returns all datasets with default settings.

  Settings: use all data (fraction=1.0); hi-realism; stable shuffling (i.e.
  independent of epoch/step and run).
  """

  ds = []
  for field_name in AllDataSets._fields:
    ds_name, ds_split = field_name.split("_")
    json_file = (
        "dataset.choy_classes.json" if ds_name == "single" else "dataset.json")
    ds_path = f"{{data_dir}}/{ds_name}.{ds_split}/{json_file}"
    ds.append(c.Dataset(
        dataset_path=ds_path, meshes_dir="{meshes_dir}", high_realism=True,
        shuffle=c.ShuffleType.ONCE, data_fraction=1.0))
  return AllDataSets(*ds)


def generate_common_string_templates() -> List[c.StringTemplate]:
  """Returns string templates common for all models."""
  return [
      # The root data directory
      c.StringTemplate(key="data_dir", value="data"),
      # Directory containing the ShapeNet meshes
      c.StringTemplate(key="meshes_dir", value="{data_dir}/shapenet_meshes"),
  ]


def generate_configs():
  common_string_templates = generate_common_string_templates()
  common_string_templates += [
      # Initial Rensnet50 checkpoint, trained on ImageNet
      c.StringTemplate(key="resnet_cpt",
                       value="{data_dir}/keras_resnet50_imagenet.cpt"),

      # Root output directory
      c.StringTemplate(key="output_dir", value="output/models")
  ]

  ds = generate_default_datasets()

  # 128^3 voxelization, fixed grid offset, FG/BG reconstruction
  # Use for training models h5, h7 and for evaluation of h5, h7, y1
  vox_fgbg_128_fixed = c.VoxelizationConfig(
      task_type=c.TaskType.FG_BG, resolution=c.Resolution(128, 128, 128),
      sub_grid_sampling=False, conservative_rasterization=False,
      voxelization_image_resolution_multiplier=8)

  # 32^3 voxelization, random grid offset, FG/BG reconstruction, improved
  # approximation thorough sub-grid sampling.
  # Use for training model y1
  vox_fgbg_32_rnd = c.VoxelizationConfig(
      task_type=c.TaskType.FG_BG, resolution=c.Resolution(32, 32, 32),
      sub_grid_sampling=True, conservative_rasterization=False,
      voxelization_image_resolution_multiplier=31)

  # 128^3 voxelization, fixed grid offset, semantic class reconstruction
  # Use for training and evaluation of models models m7 and m9
  vox_sem_128_fixed = dataclasses.replace(vox_fgbg_128_fixed,
                                          task_type=c.TaskType.SEMANTIC)

  # Training parameters common to all models
  common_train_params = dict(
      resnet50_imagenet_checkpoint="{resnet_cpt}",
      checkpoint_interval=10000,
      persistent_checkpoint_interval=500000,
      last_upscale_factor=2,
      latent_channels=64,
      skip_fraction=0.75,
      max_steps=16000000,
      tensorboard_log_interval=1000,
      initial_learning_rate=0.0004,
      adam_epsilon=0.0001,
  )

  h5 = c.TrainPipeline(
      string_templates=common_string_templates,
      train=c.TrainConfig(
          data=c.DataPipeline(
              datasets=[shuffle_per_epoch(lo_realism(ds).single_train),
                        shuffle_per_epoch(lo_realism(ds).single_val)],
              data_loader=create_data_loader(), shuffle=c.ShuffleType.PER_EPOCH,
              voxelization_config=vox_fgbg_128_fixed),
          random_grid_offset=False, **common_train_params),
      eval=create_evals(lo_realism(ds), 1, vox_fgbg_128_fixed),
      output_path="{output_dir}/h5"
  )

  h7 = c.TrainPipeline(
      string_templates=common_string_templates,
      train=c.TrainConfig(
          data=c.DataPipeline(
              datasets=[shuffle_per_epoch(ds.single_train),
                        shuffle_per_epoch(ds.single_val)],
              data_loader=create_data_loader(), shuffle=c.ShuffleType.PER_EPOCH,
              voxelization_config=vox_fgbg_128_fixed),
          random_grid_offset=False, **common_train_params),
      eval=create_evals(ds, 1, vox_fgbg_128_fixed),
      output_path="{output_dir}/h7"
  )

  y1 = c.TrainPipeline(
      string_templates=common_string_templates,
      train=c.TrainConfig(
          data=c.DataPipeline(
              datasets=[shuffle_per_epoch(ds.single_train),
                        shuffle_per_epoch(ds.single_val)],
              data_loader=create_data_loader(), shuffle=c.ShuffleType.PER_EPOCH,
              voxelization_config=vox_fgbg_32_rnd),
          random_grid_offset=True, **common_train_params),
      eval=create_evals(ds, 1, vox_fgbg_128_fixed),
      output_path="{output_dir}/y1"
  )

  m7 = c.TrainPipeline(
      string_templates=common_string_templates,
      train=c.TrainConfig(
          data=c.DataPipeline(
              datasets=[shuffle_per_epoch(ds.pairs_train),
                        shuffle_per_epoch(ds.pairs_val)],
              data_loader=create_data_loader(), shuffle=c.ShuffleType.PER_EPOCH,
              voxelization_config=vox_sem_128_fixed),
          random_grid_offset=False, **common_train_params),
      eval=create_evals(ds, 2, vox_sem_128_fixed),
      output_path="{output_dir}/m7"
  )

  m9 = c.TrainPipeline(
      string_templates=common_string_templates,
      train=c.TrainConfig(
          data=c.DataPipeline(
              datasets=[shuffle_per_epoch(ds.triplets_train),
                        shuffle_per_epoch(ds.triplets_val)],
              data_loader=create_data_loader(), shuffle=c.ShuffleType.PER_EPOCH,
              voxelization_config=vox_sem_128_fixed),
          random_grid_offset=False, **common_train_params),
      eval=create_evals(ds, 3, vox_sem_128_fixed),
      output_path="{output_dir}/m9"
  )

  config_dir = pathlib.Path(__file__).parent.parent / "configs" / "models"
  (config_dir / "h5.json5").write_text(dumps(h5))
  (config_dir / "h7.json5").write_text(dumps(h7))
  (config_dir / "m7.json5").write_text(dumps(m7))
  (config_dir / "m9.json5").write_text(dumps(m9))
  # (config_dir / "y1.json5").write_text(dumps(y1))  # y1 is still untested


def generate_paper_tf_eval_configs():
  ds = generate_default_datasets()
  common_string_templates = generate_common_string_templates()
  common_string_templates += [
      # Directory containing the pre-trained models from the paper
      c.StringTemplate("paper_tf_models_dir",
                       "{data_dir}/paper_tf_models"),

      # Root output directory
      c.StringTemplate(key="output_dir", value="output/paper_tf_models")
  ]

  vox_fgbg = c.VoxelizationConfig(
      task_type=c.TaskType.FG_BG, resolution=c.Resolution(128, 128, 128),
      sub_grid_sampling=False, conservative_rasterization=False,
      voxelization_image_resolution_multiplier=4,
      voxelization_projection_depth_multiplier=1)
  vox_h7 = dataclasses.replace(vox_fgbg,
                               voxelization_projection_depth_multiplier=2)
  vox_sem = dataclasses.replace(vox_fgbg, task_type=c.TaskType.SEMANTIC)

  default_data_loader = c.DataLoaderConfig(num_data_workers=6, batch_size=8)

  common_eval_params = dict(
      name="full_eval", num_qualitative_results=40,
      num_qualitative_results_in_tensor_board=0,
  )

  h5 = c.TfModelEvalPipeline(
      eval_config=c.EvalConfig(
          data=c.DataPipeline(
              datasets=[lo_realism(ds).single_test], shuffle=c.ShuffleType.ONCE,
              data_loader=default_data_loader, voxelization_config=vox_fgbg),
          **common_eval_params),
      frozen_graph_path="{paper_tf_models_dir}/h5.pb",
      string_templates=common_string_templates, output_path="{output_dir}/h5")
  h7 = c.TfModelEvalPipeline(
      eval_config=c.EvalConfig(
          data=c.DataPipeline(
              datasets=[ds.single_test], shuffle=c.ShuffleType.ONCE,
              data_loader=default_data_loader, voxelization_config=vox_h7),
          **common_eval_params),
      frozen_graph_path="{paper_tf_models_dir}/h7.pb",
      string_templates=common_string_templates, output_path="{output_dir}/h7")
  m7 = c.TfModelEvalPipeline(
      eval_config=c.EvalConfig(
          data=c.DataPipeline(
              datasets=[ds.pairs_test], shuffle=c.ShuffleType.ONCE,
              data_loader=default_data_loader, voxelization_config=vox_sem),
          **common_eval_params),
      frozen_graph_path="{paper_tf_models_dir}/m7.pb",
      string_templates=common_string_templates, output_path="{output_dir}/m7")
  m9 = c.TfModelEvalPipeline(
      eval_config=c.EvalConfig(
          data=c.DataPipeline(
              datasets=[ds.triplets_test], shuffle=c.ShuffleType.ONCE,
              data_loader=default_data_loader, voxelization_config=vox_sem),
          **common_eval_params),
      frozen_graph_path="{paper_tf_models_dir}/m9.pb",
      string_templates=common_string_templates, output_path="{output_dir}/m9")
  y1 = c.TfModelEvalPipeline(
      eval_config=c.EvalConfig(
          data=c.DataPipeline(
              datasets=[lo_realism(ds).single_test], shuffle=c.ShuffleType.ONCE,
              data_loader=default_data_loader, voxelization_config=vox_fgbg),
          **common_eval_params),
      frozen_graph_path="{paper_tf_models_dir}/y1.pb",
      string_templates=common_string_templates, output_path="{output_dir}/y1")
  config_dir = (pathlib.Path(__file__).parent.parent /
                "configs" / "paper_tf_models")
  config_dir.mkdir(parents=True)
  (config_dir / "h7.json5").write_text(dumps(h7))
  (config_dir / "h5.json5").write_text(dumps(h5))
  (config_dir / "m7.json5").write_text(dumps(m7))
  (config_dir / "m9.json5").write_text(dumps(m9))
  (config_dir / "y1.json5").write_text(dumps(y1))


def generate_schemas():
  schema_dir = pathlib.Path(__file__).parent.parent / "configs" / "schemas"
  schema_dir.mkdir(parents=True, exist_ok=True)

  (schema_dir / "tf_model_eval_config.json").write_text(json.dumps(
      c.TfModelEvalPipeline.json_schema(), sort_keys=True,
      indent=2))
  (schema_dir / "dataset_config.json").write_text(json.dumps(
      dataset_lib.DatasetConfig.json_schema(), sort_keys=True, indent=2))
  (schema_dir / "train_config.json").write_text(json.dumps(
      c.TrainPipeline.json_schema(), sort_keys=True, indent=2))


def main():
  generate_schemas()
  generate_configs()
  generate_paper_tf_eval_configs()


if __name__ == '__main__':
  main()
