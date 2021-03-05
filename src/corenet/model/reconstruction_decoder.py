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

"""3D reconstruction decoder."""
import collections

import numpy as np
import torch as t
from torch import nn

import corenet.configuration as configuration
from corenet.geometry import transformations
from corenet.model import batch_renorm
from corenet.model import ray_traced_skip_connection
from corenet.model import resnet50


class ReconstructionDecoder(nn.Module):
  """Base class for voxel grid decoders starting from ResNet50 features."""

  def __init__(self, config: configuration.DecoderConfig):
    """Initializes the decoder."""
    super().__init__()

    self.config = config

    depth, height, width = config.resolution
    div = 16 * config.last_upscale_factor
    assert depth % div == 0 and height % div == 0 and width % div == 0
    # Resolution for the first grid after the latent layer
    initial_grid_resolution = (depth // div, height // div, width // div)

    bn = lambda v: batch_renorm.BatchRenorm(v, eps=0.001)
    rt_skip = ray_traced_skip_connection.SampleGrid2d
    relu, conv3d, conv3d_t = nn.ReLU, nn.Conv3d, nn.ConvTranspose3d
    ir = np.array(initial_grid_resolution)

    self.stage_0 = nn.Linear(2048, config.latent_channels)

    self.stage_1 = nn.Sequential(collections.OrderedDict(
        r1=relu(), b1=bn(config.latent_channels + 3),
        t1=conv3d_t(config.latent_channels + 3, 256, 4,
                    stride=initial_grid_resolution)))

    self.stage_2 = nn.Sequential(collections.OrderedDict(
        r1=relu(), b1=bn(256), c1=conv3d(256, 256, 3, padding=1),
        r2=relu(), b2=bn(256),
        t1=conv3d_t(256, 128, 3, stride=2, padding=1, output_padding=1)
    ))
    skip2_channels = round(128 * config.skip_fraction)
    self.rt_skip_2 = rt_skip(2048 + 3, skip2_channels, ir * 2)

    in3c = 128 + skip2_channels
    self.stage_3 = nn.Sequential(collections.OrderedDict(
        r1=relu(), b1=bn(in3c), c1=conv3d(in3c, 128, 5, padding=2),
        r2=relu(), b2=bn(128),
        t1=conv3d_t(128, 64, 7, stride=2, padding=3, output_padding=1)))
    skip3_channels = round(64 * config.skip_fraction)
    self.rt_skip_3 = rt_skip(1024 + 3, skip3_channels, ir * 4)

    in4c = 64 + skip3_channels
    self.stage_4 = nn.Sequential(collections.OrderedDict(
        r1=relu(), b1=bn(in4c), c1=conv3d(in4c, 64, 5, padding=2),
        r2=relu(), b2=bn(64),
        t1=conv3d_t(64, 32, 7, stride=2, padding=3, output_padding=1)))
    skip4_channels = round(32 * config.skip_fraction)
    self.rt_skip_4 = rt_skip(512 + 3, skip4_channels, ir * 8)

    in5c = 32 + skip4_channels
    self.stage_5 = nn.Sequential(collections.OrderedDict(
        r1=relu(), b1=bn(in5c), c1=conv3d(in5c, 32, 5, padding=2),
        r2=relu(), b2=bn(32),
        t1=conv3d_t(32, 16, 7, stride=2, padding=3, output_padding=1)
    ))
    skip5_channels = round(16 * config.skip_fraction)
    self.rt_skip_5 = rt_skip(256 + 3, skip5_channels, ir * 16)

    in6c = 16 + skip5_channels
    self.stage_6 = nn.Sequential(collections.OrderedDict(
        r1=relu(), b1=bn(in6c), c1=conv3d(in6c, 16, 5, padding=2),
        r2=relu(), b2=bn(16),
        t1=conv3d_t(16, config.num_output_channels, 7,
                    stride=config.last_upscale_factor, padding=3,
                    output_padding=1)))

  def _apply_skip(
      self, src3d: t.Tensor, src2d: t.Tensor, stage: int,
      voxel_projection_matrix: t.Tensor,
      voxel_sample_locations: t.Tensor
  ) -> t.Tensor:
    skip_fn = getattr(
        self, f"rt_skip_{stage}", None
    )  # type: ray_traced_skip_connection.SampleGrid2d

    if not skip_fn:
      return src3d
    o = voxel_sample_locations[:, :, None, None]
    o = o.expand(src2d.shape[0], o.shape[1], *src2d.shape[2:])
    src2d = t.cat([src2d, o], 1)
    r1 = src3d.new_tensor(src3d.shape[2:], dtype=t.float32)
    r2 = src3d.new_tensor(self.config.resolution, dtype=t.float32)
    layer_scale = transformations.scale(r2 / r1)
    layer_matrix = voxel_projection_matrix.matmul(
        layer_scale.to(voxel_projection_matrix.device))
    skip_activations = skip_fn(src2d, layer_matrix, voxel_sample_locations)
    return t.cat([src3d, skip_activations], dim=1)

  def forward(self, image_features: resnet50.ResNet50Features,
              voxel_projection_matrix: t.Tensor,
              voxel_sample_locations: t.Tensor):
    """Forward pass.

    Args:
      image_features: The image features, returned by ResNet50.
      voxel_projection_matrix: Voxel to screen space transformation matrix to
        use for the skip connections, float32[batch_size, 4, 4] or None
      voxel_sample_locations: The locations at which the voxels were sampled,
        float32[batch_size, 3]

    Returns:
      The predicted grid logits.
    """

    matrices = [voxel_projection_matrix, voxel_sample_locations]

    imf = image_features
    x = self.stage_0(imf.global_average_2048)
    x = t.cat([x, voxel_sample_locations], 1)
    x = x[:, :, None, None, None]
    x = self.stage_1(x)
    x = self._apply_skip(x, imf.stage5_2048x8x8, 1, *matrices)
    x = self.stage_2(x)
    x = self._apply_skip(x, imf.stage5_2048x8x8, 2, *matrices)
    x = self.stage_3(x)
    x = self._apply_skip(x, imf.stage4_1024x16x16, 3, *matrices)
    x = self.stage_4(x)
    x = self._apply_skip(x, imf.stage3_512x32x32, 4, *matrices)
    x = self.stage_5(x)
    x = self._apply_skip(x, imf.stage2_256x64x64, 5, *matrices)
    x = self.stage_6(x)
    return x
