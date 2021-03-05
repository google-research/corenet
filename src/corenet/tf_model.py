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

"""Evaluates a frozen CoreNet TensorFlow graph."""

import logging

import numpy as np
import re
import tensorflow as tf
import torch as t

from corenet import file_system as fs
from corenet import misc_util
from corenet import super_resolution as super_resolution_lib

log = logging.getLogger(__name__)


class TfFrozenGraphModel(super_resolution_lib.MultiOffsetInferenceFn):
  def __init__(self, graph_path: str):
    tf1 = tf.compat.v1
    graph_def = tf1.GraphDef.FromString(fs.read_bytes(graph_path))
    resolution_node = [
        v.attr["value"] for v in graph_def.node
        if v.name == "output_resolution"][0].tensor
    assert resolution_node.dtype == tf.int32.as_datatype_enum
    self.output_shape = tuple(
        np.frombuffer(resolution_node.tensor_content, dtype=np.int32))

    @tf.function
    def _call_tf_graph(input_image, camera_transform, view_to_voxel_transform,
                       grid_offset) -> tf.Tensor:
      input_map = {
          "input_image": input_image,
          "camera_transform": camera_transform,
          "grid_offset": grid_offset,
          "view_to_voxel_transform": view_to_voxel_transform
      }

      pmf, = tf1.import_graph_def(graph_def, input_map=input_map,
                                  return_elements=["class_pdf:0"])
      return pmf

    self._call_tf_graph = _call_tf_graph

  def _run_inference(
      self, input_image: tf.Tensor, camera_transform: tf.Tensor,
      view_to_voxel_transform: tf.Tensor, grid_offsets: tf.Tensor
  ) -> tf.Tensor:
    num_offsets = grid_offsets.shape[0]
    pmfs = tf.TensorArray(tf.float32, size=num_offsets)
    for idx in range(num_offsets):
      grid_offset = grid_offsets[idx]
      pmf = self._call_tf_graph(
          input_image, camera_transform, view_to_voxel_transform, grid_offset)
      pmfs = pmfs.write(idx, pmf)
    return pmfs.stack()

  def __call__(
      self, input_image: t.Tensor, camera_transform: t.Tensor,
      view_to_voxel_transform: t.Tensor, grid_offsets: t.Tensor
  ) -> t.Tensor:
    assert len(grid_offsets.shape) == 3
    # Torch->TF: Permute to NHWC and convert to a numpy array.
    result_device = input_image.device
    input_image = tf.convert_to_tensor(
        input_image.to(t.float32).permute([0, 2, 3, 1]).cpu().numpy())
    camera_transform = tf.convert_to_tensor(
        camera_transform.cpu().numpy())
    view_to_voxel_transform = tf.convert_to_tensor(
        view_to_voxel_transform.cpu().numpy())
    grid_offsets = tf.convert_to_tensor(grid_offsets.cpu().numpy())

    pmfs = self._run_inference(
        input_image, camera_transform, view_to_voxel_transform, grid_offsets)
    pmfs = misc_util.to_tensor(pmfs.numpy(), t.float32)
    # TF->Torch: Permute to NCHWD and store on same device as input.
    pmfs = pmfs.to(result_device).permute([0, 1, 5, 2, 3, 4])
    return pmfs


def setup_tensorflow(gpu_index: int):
  """Make sure TF uses same GPU as torch and does not consume all memory."""
  physical_devices = {
      int(m.group(1)): v
      for v in tf.config.list_physical_devices("GPU")
      if (m := re.match(r"/physical_device:GPU:(\d+)", v.name))
  }

  if gpu_index not in physical_devices:
    raise ValueError(
        f"GPU {gpu_index} is visible to PyTorch but not TensorFlow. "
        "Use CUDA_VISIBLE_DEVICES to ignore it before starting the program.")
  tf_gpu = physical_devices[gpu_index]
  tf.config.set_visible_devices([tf_gpu], "GPU")
  tf.config.experimental.set_memory_growth(tf_gpu, True)
  assert tf.executing_eagerly()


def super_resolution_from_tf_model(graph_path: str):
  tf_model = TfFrozenGraphModel(graph_path)
  return super_resolution_lib.SuperResolutionInference(
      tf_model, tf_model.output_shape[:-1])
