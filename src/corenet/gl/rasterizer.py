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

"""An OpenGL rasterizer for PyTorch."""

import dataclasses
import importlib
import logging
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Union

import OpenGL.GL
import OpenGL.GL.NV.conservative_raster
import glcontext
import moderngl
import numpy as np
import torch as t

from corenet.gl import egl_context

importlib.reload(glcontext)
egl_context.monkey_patch_moderngl()

InputTensor = Union[t.Tensor, np.ndarray, int, float, bool, Iterable]

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Uniform:
  name: str
  value: InputTensor


@dataclasses.dataclass()
class Buffer:
  binding: int
  value: InputTensor
  is_io: bool = False


@dataclasses.dataclass(frozen=True)
class Texture:
  name: str
  value: InputTensor
  bind_as_array: bool = False


@dataclasses.dataclass(frozen=True)
class RenderInput:
  # The number of points to render. The geometry shader can then convert these
  # into other geometry.
  num_points: int

  # The parameters to pass to the shaders.
  arguments: Iterable[Union[Uniform, Buffer, Texture]]

  # The vertex shader
  vertex_shader: str

  # The geometry shader
  geometry_shader: str

  # The fragment shader
  fragment_shader: str

  # The output resolution, tuple (height, width)
  output_resolution: Tuple[int, int] = (256, 256)

  # The clear color, tuple (R, G, B) or (R, G, B, A).
  clear_color: Iterable[float] = (0, 0, 0)

  # The output type (either uint8 or float32).
  output_type: t.dtype = t.uint8

  # Whether depth testing is enabled.
  depth_test_enabled: bool = True

  # Whether to use conservative rasterization
  conservative_rasterization: bool = False


class _EglRenderer:
  _context_cache = {}  # type: Dict[int, _EglRenderer]

  def __init__(self, gl_context: moderngl.Context):
    self._program_cache = {}  # type: Dict[str, moderngl.Program]
    self._gl_context = gl_context

  @classmethod
  def get_instance(cls, cuda_device: int):
    if cuda_device not in cls._context_cache:
      # Cuda has to be initialized for the cuda_egl backend to work
      t.cuda.init()
      ctx = moderngl.create_standalone_context(backend="cuda_egl",
                                               cuda_device=cuda_device)
      cls._context_cache[cuda_device] = _EglRenderer(ctx)
    return cls._context_cache[cuda_device]

  def _check_gl_error(self):
    err = self._gl_context.error
    if err != "GL_NO_ERROR":
      raise ValueError(f"OpenGL error encountered: {err}")

  def _upload_uniform(self, program: moderngl.Program,
                      parameter: Uniform):
    if parameter.name not in program:
      log.info(f"Uniform {parameter.name} not found in program")
    else:
      value = t.as_tensor(parameter.value).cpu()
      if len(value.shape) == 2:
        value = tuple(value.transpose(0, 1).reshape([-1]))
      elif len(value.shape) == 1:
        value = tuple(value)
      elif value.shape == ():
        value = value.item()
      else:
        raise ValueError("Only supports 0, 1, and 2 dim tensors.")
      program[parameter.name].value = value

  def render(self, render_input: RenderInput):
    inp = render_input
    with self._gl_context:
      program_key = (
          f"{inp.vertex_shader}|{inp.geometry_shader}|{inp.fragment_shader}")

      if program_key not in self._program_cache:
        self._program_cache[program_key] = self._gl_context.program(
            vertex_shader=inp.vertex_shader,
            fragment_shader=inp.fragment_shader,
            geometry_shader=inp.geometry_shader)
      program = self._program_cache[program_key]

      objects_to_delete = []
      buffer_bindings = {}  # type: Dict[int, moderngl.Buffer]
      texture_location = 0
      try:
        for parameter in inp.arguments:
          if isinstance(parameter, Uniform):
            self._upload_uniform(program, parameter)
          elif isinstance(parameter, Buffer):
            value = t.as_tensor(parameter.value).reshape([-1]).cpu().numpy()
            buffer = self._gl_context.buffer(value)
            buffer.bind_to_storage_buffer(parameter.binding)
            buffer_bindings[parameter.binding] = buffer
            objects_to_delete.append(buffer)
          elif isinstance(parameter, Texture):
            if not parameter.bind_as_array:
              raise NotImplementedError()
            if parameter.name not in program:
              log.info(f"Uniform {parameter.name} not found in program")
            else:
              val = parameter.value.cpu().numpy()
              texture = self._gl_context.texture_array(
                  val.shape[1:3] + val.shape[0:1], val.shape[3], val)
              texture.repeat_x = True
              texture.repeat_y = True
              texture.use(location=texture_location)
              program.get(parameter.name, None).value = texture_location
              texture_location += 1
              objects_to_delete.append(texture)
          else:
            raise ValueError("Unknown parameter type")
          self._check_gl_error()

        h, w = inp.output_resolution
        gl_dtype, np_dtype = {
            t.uint8: ("f1", np.uint8),
            t.float32: ("f4", np.float32)
        }[inp.output_type]

        render_buffer = self._gl_context.renderbuffer(
            (w, h), components=4, samples=0, dtype=gl_dtype)
        objects_to_delete.append(render_buffer)
        depth_buffer = self._gl_context.depth_renderbuffer((w, h), samples=0)
        objects_to_delete.append(depth_buffer)
        framebuffer = self._gl_context.framebuffer(render_buffer, depth_buffer)
        objects_to_delete.append(framebuffer)
        framebuffer.use()

        vertex_array = self._gl_context.vertex_array(program, ())
        objects_to_delete.append(vertex_array)
        self._gl_context.clear(*inp.clear_color)
        self._gl_context.disable(moderngl.CULL_FACE)
        if inp.depth_test_enabled:
          self._gl_context.enable(moderngl.DEPTH_TEST)
          self._gl_context.depth_func = "<="
        else:
          self._gl_context.disable(moderngl.DEPTH_TEST)

        if render_input.conservative_rasterization:
          OpenGL.GL.glEnable(
              OpenGL.GL.NV.conservative_raster.GL_CONSERVATIVE_RASTERIZATION_NV)
        else:
          OpenGL.GL.glDisable(
              OpenGL.GL.NV.conservative_raster.GL_CONSERVATIVE_RASTERIZATION_NV)

        vertex_array.render(mode=moderngl.POINTS, vertices=inp.num_points)
        self._check_gl_error()
        result = np.zeros([h, w, 4], dtype=np_dtype)
        framebuffer.read_into(result, components=4, dtype=gl_dtype)
        self._check_gl_error()

        for parameter in inp.arguments:
          if isinstance(parameter, Buffer) and parameter.is_io:
            old = parameter.value
            npdt = {t.float32: np.float32, t.int32: np.int32}[old.dtype]
            temp_buffer = np.zeros(old.shape, npdt)
            buffer_bindings[parameter.binding].read_into(temp_buffer)
            parameter.value = t.as_tensor(temp_buffer)

        return t.as_tensor(result)

      finally:
        for v in objects_to_delete:
          v.release()


def gl_simple_render(
    render_input: RenderInput,
    cuda_device: Optional[int] = None
):
  """Renders the supplied configuration with OpenGL.

  Args:
    render_input: The render input
    cuda_device: The GPU to use, given as a CUDA device number

  Returns:
    The rendered image, output_type[height, width, 4]

  Buffer parameters with is_io set to True will be read back. This can either
  happen in the original buffer or in a new buffer. The implementation will
  update the value field in the buffer parameter in the latter case.

  There is no guarantee about the device of both the rendered image and the
  buffers that are read back.
  """
  if cuda_device is None:
    cuda_device = t.cuda.current_device()
  instance = _EglRenderer.get_instance(cuda_device)
  return instance.render(render_input)
