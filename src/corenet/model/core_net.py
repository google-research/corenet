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

"""CoreNet image to 3D reconstruction network."""

import torch as t
from torch import nn

import corenet.configuration as configuration
from corenet.model import reconstruction_decoder
from corenet.model import resnet50


class CoreNet(nn.Module):
  """Image to 3D reconstruction with CoreNet."""

  def __init__(self, config: configuration.CoreNetConfig):
    """Initializes the module."""
    super().__init__()

    self.config = config
    self.encoder = resnet50.ResNet50FeatureExtractor()
    self.decoder = reconstruction_decoder.ReconstructionDecoder(config.decoder)

  def forward(self, image: t.Tensor,
              voxel_projection_matrix: t.Tensor,
              voxel_sample_locations: t.Tensor) -> t.Tensor:
    """The forward pass (see documentation of __call__)."""
    image = resnet50.preprocess_image_caffe(image)
    features = self.encoder(image)
    return self.decoder.forward(features, voxel_projection_matrix,
                                voxel_sample_locations)

  def __call__(self, image: t.Tensor,
               voxel_projection_matrix: t.Tensor,
               voxel_sample_locations: t.Tensor) -> t.Tensor:
    """The forward pass.

    Args:
      image: The input image, float32[batch_size, 3, height, width]
      voxel_projection_matrix: Voxel to screen space transformation matrix to
        use for the skip connections, float32[batch_size, 4, 4] or None
      voxel_sample_locations: The locations at which the voxels were sampled,
        float32[batch_size, 3]

    Returns:
      The predicted grid logits.
    """
    return super().__call__(
        image, voxel_projection_matrix, voxel_sample_locations)
