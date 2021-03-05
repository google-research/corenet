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

"""Library for filling isolated inner regions of objects in a voxel grid.

Uses connected component analysis to fill all empty inner regions with ones.
Regions are defined with 8-way connectivity in 3D. All voxels of an empty region
are zeros.

There are two implementations provided (GPU and CPU), implemented as extensions
in Cuda/C++. The extensions will be JIT compiled from sources, if missing.
You need to have a valid CUDA and gcc installations.
Use the CUDA_HOME environment variable to specify the CUDA home. You can pass
semi-colon separated extra options to nvcc, using the FILL_VOXELS_CUDA_FLAGS
environment variable.

Invoking this module as main will compile the extension (ahead of time) and
store it in torch's extension dir. To change the output location, set the
TORCH_EXTENSIONS_DIR environment variable.
"""

import importlib.machinery
import importlib.util
import logging
from importlib import resources
from typing import Iterable
from typing import Optional
from typing import Union

import numpy as np
import os
import torch as t
import types
from torch.utils import cpp_extension

from corenet import cc

InputTensor = Union[t.Tensor, np.ndarray, int, float, Iterable]

_corenet_cpp: Optional[types.ModuleType] = None

# Linking errors related to parallel_for could be related to torch library
# being built with a different threading model. Try different values for
# _ATEN_THREADING (see torch C++ code).
_ATEN_THREADING = "OPENMP"

log = logging.getLogger(__name__)


def get_module(verbose=False):
  """Returns a module

  Args:
    verbose:

  Returns:

  """
  global _corenet_cpp
  force_verbose = int(os.environ.get("CORENET_VERBOSE_CC_COMPILE", "-1"))
  if force_verbose >= 0:
    verbose = bool(force_verbose)
  if not _corenet_cpp:
    module_path = os.environ.get("CORENET_PRECOMPILED_CPP_MODULE_PATH", None)
    module_name = "corenet_cpp"
    if module_path:
      log.debug(f"Loading flood fill routine from {module_path}")
      module_path = os.path.join(module_path, f"{module_name}/{module_name}.so")
      spec = importlib.util.spec_from_file_location(module_name, module_path)
      _corenet_cpp = importlib.util.module_from_spec(spec)
    else:
      log.debug(f"JIT compiling flood fill routine from sources... (can take a while on the first run)")
      extra_cuda_flags = os.environ.get("FILL_VOXELS_CUDA_FLAGS", "")
      extra_cuda_flags = list(filter(None, extra_cuda_flags.split(";")))
      with resources.path(cc, "fill_voxels_cpu.cc") as p1, \
          resources.path(cc, "fill_voxels_gpu.cu") as p2, \
          resources.path(cc, "module.cc") as p3:
        _corenet_cpp = cpp_extension.load(
            name=module_name, sources=[str(p1), str(p2), str(p3)],
            verbose=verbose,
            with_cuda=True,
            extra_cuda_cflags=(
                ["-std=c++14", "-O2", "-Xcudafe=--display_error_number"]
                + extra_cuda_flags),
            extra_cflags=["-std=c++17", "-fsized-deallocation", "-O2",
                          f"-DAT_PARALLEL_{_ATEN_THREADING}"])
      log.debug(f"Finished JIT compiling")
  return _corenet_cpp


def fill_inside_voxels_cpu(grid: t.Tensor) -> t.Tensor:
  return get_module().fill_inside_voxels_cpu(grid)


def fill_inside_voxels_gpu(grid: t.Tensor, inplace=False) -> t.Tensor:
  return get_module().fill_inside_voxels_gpu(grid, inplace)


if __name__ == '__main__':
  get_module(verbose=True)
