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

#pragma diag_suppress 68, 2361, 3058, 3060
#include <cassert>
#include <cuda.h>
#include <torch/torch.h>
#include <pybind11/pybind11.h>
#include <ATen/cuda/CUDAContext.h>

constexpr const int ROOT_MARKER = 0x7FFF'FFFF;

namespace {
__device__ int find_root(int node, int *__restrict__ parent_array) {
  int first_node = node;
  for (int l = 0;; l++) {
    int p = parent_array[node];
    if (p == ROOT_MARKER) return node;
    parent_array[first_node] = p;
    node = p;
  }
}

__device__ void merge_trees(int a, int b, int *__restrict__ parent_array) {
  for (;;) {
    a = find_root(a, parent_array);
    b = find_root(b, parent_array);

    if (a == b) break;

    int aa = a, bb = b;
    a = min(aa, bb);
    b = max(aa, bb);
    // Attach the node with the larger address under the node with the smaller.
    // This way background (label 0) always remains root and can be used to
    // reliably detect outside regions below.
    // Race conditions in the atomicMean are also reduced.
    int old_value = atomicMin(parent_array + b, a);
    if (old_value == ROOT_MARKER) break;
    b = old_value;
  };
}

constexpr dim3 BLOCK_SIZE = {8, 4, 4};

template<class T>
using DataAccessor = at::PackedTensorAccessor32<T, 4, at::RestrictPtrTraits>;

template<class T>
__forceinline__ __device__ int4 get_size(DataAccessor<T> data) {
  return int4{
      .x=data.size(3), .y=data.size(2), .z=data.size(1), .w=data.size(0)};
}

__forceinline__ __device__ int4 get_position(int4 size) {
  int x = blockIdx.x * BLOCK_SIZE.x + threadIdx.x;
  int y = blockIdx.y * BLOCK_SIZE.y + threadIdx.y;
  int z = blockIdx.z * BLOCK_SIZE.z + threadIdx.z;
  int b = z / size.z;
  z %= size.z;
  return int4{.x=x, .y=y, .z=z, .w=b};
}


// Returns the node corresponding to a position. Node 0 is reserved for BG,
// so all nodes are offset by 1.
__forceinline__ __device__ int get_node(int4 p, int4 s) {
  return ((p.w * s.z + p.z) * s.y + p.y) * s.x + p.x + 1;
}

// Checks if a position is inside the volume. Returns 1, if it is; 0 if
// it is outside but can still influence the regions of the volume through
// a tree merge operation. And -1 otherwise (outside, cannot influence).
__forceinline__ __device__ int is_inside(int4 p, int4 s) {
  if (p.x >= s.x || p.y >= s.y || p.z >= s.z || p.w >= s.w) return -1;
  if (p.x == s.x || p.y == s.y || p.z == s.z) {
    bool at_corner = (p.x == s.x && p.y == s.y) || (p.x == s.x && p.z == s.z)
        || (p.y == s.y && p.z == s.z);
    return at_corner ? -1 : 0;
  }
  return 1;

}

template<class T>
__global__ void merge_neighbour_regions(
    int *__restrict__ parent_array, DataAccessor<T> data) {
  int4 s = get_size(data);
  int4 p = get_position(s);
  int inside = is_inside(p, s);
  if (inside < 0) return;

  int node = get_node(p, s);

  bool value_cur = inside > 0 ? data[p.w][p.z][p.y][p.x] > 0 : false;
  int node_cur = inside > 0 ? node : 0;

  T value_left = p.x > 0 ? data[p.w][p.z][p.y][p.x - 1] > 0 : false;
  int node_left = p.x > 0 ? node - 1 : 0;
  if (value_cur == value_left) merge_trees(node_cur, node_left, parent_array);

  T value_top = p.y > 0 ? data[p.w][p.z][p.y - 1][p.x] > 0 : false;
  int node_top = p.y > 0 ? node - s.x : 0;
  if (value_cur == value_top) merge_trees(node_cur, node_top, parent_array);

  T value_front = p.z > 0 ? data[p.w][p.z - 1][p.y][p.x] > 0 : false;
  int node_front = p.z > 0 ? node - s.x * s.y : 0;
  if (value_cur == value_front) merge_trees(node_cur, node_front, parent_array);
}

template<class T>
__global__ void compress_paths(int *parent_array, DataAccessor<T> result) {
  int4 s = get_size(result);
  int4 p = get_position(s);
  int inside = is_inside(p, s);
  if (inside <= 0) return;
  int node = get_node(p, s);

  int root = find_root(node, parent_array);
  result[p.w][p.z][p.y][p.x] = root == 0 ? 0 : 1;
}

} // namespace

at::Tensor fill_inside_voxels_gpu(at::Tensor grid, bool inplace = false) {
  if (!grid.device().is_cuda()) {
    throw pybind11::value_error("Only CUDA tensors are supported by this OP");
  }

  const auto &shape = grid.sizes();
  if (shape.size() != 4) {
    throw pybind11::value_error("Expecting rank 4 tensor");
  }

  at::Tensor parent_array = grid.new_full(
      {4 * grid.numel() + 4}, ROOT_MARKER,
      at::TensorOptions().dtype(torch::kI32));

  dim3 grid_size = {
      (unsigned) (grid.size(3) + 1 + BLOCK_SIZE.x - 1) / BLOCK_SIZE.x,
      (unsigned) (grid.size(2) + 1 + BLOCK_SIZE.y - 1) / BLOCK_SIZE.y,
      (unsigned) ((grid.size(1) + 1) * grid.size(0) + BLOCK_SIZE.z - 1)
          / BLOCK_SIZE.z
  };

  AT_DISPATCH_ALL_TYPES(grid.scalar_type(), "fill_inside_voxels_gpu", ([&] {
    merge_neighbour_regions<scalar_t><<<grid_size, BLOCK_SIZE>>>(
        parent_array.data_ptr<int>(),
        grid.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>());
  }));
  at::Tensor result = inplace ?
                      grid : grid.clone(c10::MemoryFormat::Contiguous);
  AT_DISPATCH_ALL_TYPES(result.scalar_type(), "fill_inside_voxels_gpu", ([&] {
    compress_paths<scalar_t><<<grid_size, BLOCK_SIZE>>>(
        parent_array.data_ptr<int>(),
        result.packed_accessor32<scalar_t, 4, at::RestrictPtrTraits>());
  }));

  return result;
}

