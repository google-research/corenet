// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/torch.h>
#include <pybind11/pybind11.h>

at::Tensor fill_inside_voxels_gpu(at::Tensor grid, bool inplace = false);
torch::Tensor fill_inside_voxels_cpu(const torch::Tensor grid);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fill_inside_voxels_gpu",
        &fill_inside_voxels_gpu,
        "Fill inside regions in a grid, using the GPU.");

  m.def("fill_inside_voxels_cpu",
        &fill_inside_voxels_cpu,
        "Fill inside regions in a grid, using the CPU.");
}



