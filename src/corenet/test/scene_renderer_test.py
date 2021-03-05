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

import math
import unittest
from importlib import resources
from typing import Text

import PIL.Image
import io
import numpy as np
import numpy.testing as tt
import torch as t

from corenet.geometry import transformations
from corenet.visualization import scene_renderer
from corenet.visualization import test_data


class SceneRendererTests(unittest.TestCase):

  def testRendersSimpleMesh(self):
    mesh_vertices = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
                              [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                             dtype=np.float32)
    mesh_indices = np.array([
        0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 7, 6, 5, 5, 4, 7, 4, 0, 3, 3, 7, 4,
        4, 5, 1, 1, 0, 4, 3, 2, 6, 6, 7, 3
    ])
    mesh = mesh_vertices[mesh_indices].reshape([-1, 3, 3])

    look_at = transformations.look_at_rh((-1.2, -1.5, -1), (0, 0, 0), (0, 1, 0))
    perspective = transformations.perspective_rh(70 * math.pi / 180, 1, 0.1,
                                                 10.0)
    camera_matrix = np.matmul(perspective, look_at)

    image = scene_renderer.render_scene(
        mesh,
        camera_matrix,
        (256, 256),
        material_ids=t.zeros(mesh.shape[0], dtype=t.int32),
        diffuse_coefficients=((1.0, 0.0, 0.0),),
        ambient_coefficients=((0.0, 0.0, 0.0),),
        ambient_light_color=(0.0, 0.0, 0.0),
        light_position=(-1.2, -1.5, -1),
    )
    image = image.numpy()

    with resources.open_binary(test_data, "expected_image_mesh.png") as fl:
      expected_image = np.array(PIL.Image.open(fl))[..., :3]

    self.assertEqual(image.dtype, np.uint8)
    self.assertEqual(tuple(image.shape), tuple(expected_image.shape))
    difference_l1 = np.abs(
        image.astype(np.int64) - expected_image.astype(np.int64)).sum()
    self.assertAlmostEqual(difference_l1, 0, 1024)

  def testDecodesTextureImages(self):
    def read_image(name: Text):
      encoded_image = resources.read_binary(test_data, name)
      pil_image = (
          PIL.Image.open(io.BytesIO(encoded_image)))  # type: PIL.Image.Image
      pil_image = pil_image.convert("RGB").resize((192, 128),
                                                  resample=PIL.Image.BICUBIC)
      image = t.as_tensor(np.array(pil_image)).flip(0).contiguous()
      return image, encoded_image

    image_1, encoded_1 = read_image("expected_image_mesh.png")
    image_2, encoded_2 = read_image("expected_image_voxels.png")

    texture_array, image_indices = scene_renderer.load_textures(
        [encoded_1, encoded_2, b"", encoded_1], (128, 192))

    self.assertEqual(tuple(texture_array.shape), (2, 128, 192, 3))

    tt.assert_allclose(texture_array[image_indices[0]], image_1, atol=1e-7)
    tt.assert_allclose(texture_array[image_indices[1]], image_2, atol=1e-7)
    self.assertEqual(image_indices[2].numpy(), -1)
    tt.assert_allclose(texture_array[image_indices[3]], image_1, atol=1e-7)


if __name__ == "__main__":
  unittest.main()
