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

from typing import Union

import PIL.Image
import base64
import io
import numpy as np
import re
import torch as t


class DisplayDataObject:
  plot_index = 0

  def __init__(self, image):
    self.plot_index = DisplayDataObject.plot_index
    DisplayDataObject.plot_index = -1
    self.image_width = image.shape[1]
    self.image_bytes = image.tobytes()

  def _repr_display_(self):
    image_bytes_base64 = base64.b64encode(self.image_bytes)
    image_bytes_base64 = image_bytes_base64.decode()
    body = {
        'plot_index': self.plot_index,
        'image_width': self.image_width,
        'image_base64': image_bytes_base64
    }
    return 'pycharm-plot-image', body


InputImage = Union[bytes, np.ndarray, t.Tensor]


def ij_display(image: InputImage):
  """Displays an image in IntelliJ's SciView."""
  if isinstance(image, bytes):
    image = np.array(PIL.Image.open(io.BytesIO(image)))
  if t.is_tensor(image):
    image = image.detach().cpu().numpy()
  image = np.asarray(image)
  if len(image.shape) == 3 and image.shape[-1] == 1:
    image = image.squeeze(-1)
  if len(image.shape) == 2:
    image = np.stack([image] * 3, -1)
  assert len(image.shape) == 3
  if image.shape[-1] == 4:
    image = image[..., :3]
  assert image.shape[-1] == 3
  if image.dtype == np.float32:
    image = (image * 255).clip(0, 255).astype(np.uint8)
  assert image.dtype == np.uint8

  # noinspection PyUnresolvedReferences
  from datalore import display
  display.display(DisplayDataObject(image))


def print_tensor(v: t.Tensor):
  v = v.detach()
  dtype = re.sub(r"^torch\.", "", str(v.dtype))
  sep = "\n" if len(v.shape) > 1 else " "
  return f"{dtype}{list(v.shape)}({v.device}){{{sep}{v.cpu().numpy()}{sep}}}"


def better_tensor_display():
  """Better string representation of tensors for python debuggers."""
  np.set_printoptions(4, suppress=True)
  t.set_printoptions(4, sci_mode=False)
  t.Tensor.__repr__ = print_tensor
