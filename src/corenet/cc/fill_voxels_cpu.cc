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

#include <ATen/Parallel.h>
#include <boost/container/small_vector.hpp>
#include <pybind11/pybind11.h>
#include <torch/torch.h>
#include <tuple>
#include <vector>

namespace {

using int64 = std::int64_t;
template<class T, size_t S>
using InlinedVector = boost::container::small_vector<T, S>;

inline int64 FlatIndex(int x, int y, int z,
                       std::tuple<int, int, int> grid_size) {
  auto[size_z, size_y, size_x] = grid_size;
  return (z * size_y + y) * size_x + x;
}

// A disjoint union set that's tracking the info for key groups.
class DisjointUnionSet {
 public:
  void Merge(int64 r1, int64 r2) {
    r1 = FindRoot(r1);
    r2 = FindRoot(r2);
    if (r1 < r2) {
      parent_[r2] = r1;
    } else if (r2 < r1) {
      parent_[r1] = r2;
    }
  }

  // Finds the root of the set this key belongs to, and compress the path by
  // attaching every node along the path to the root. This operation is
  // close to O(1) on average.
  int64 FindRoot(int64 element) {
    InlinedVector<int64, 128> visited;
    int64 root = element;
    while (parent_[root] != -1) {
      visited.push_back(root);
      root = parent_[root];
    }
    for (int64 v : visited) {
      parent_[v] = root;
    }

    return root;
  }

  int64 NewRegion() {
    parent_.push_back(-1);
    return parent_.size() - 1;
  }

 private:
  // Parent node for each key.
  std::vector<int64> parent_;
};

template<class T>
void ComputeConnectedComponents(
    std::tuple<int, int, int> grid_size, const T *volume, T outside_voxels_value, int64 *regions) {
  auto[size_z, size_y, size_x] = grid_size;

  DisjointUnionSet union_set;
  int64 outside_region = union_set.NewRegion();
  CHECK_EQ(outside_region, 0);

  for (int z = 0; z < size_z; z++) {
    for (int y = 0; y < size_y; y++) {
      for (int x = 0; x < size_x; x++) {
        int64 offset_current = FlatIndex(x, y, z, grid_size);
        int64 offset_left = FlatIndex(x - 1, y, z, grid_size);
        int64 offset_back = FlatIndex(x, y - 1, z, grid_size);
        int64 offset_up = FlatIndex(x, y, z - 1, grid_size);

        bool value_current = volume[offset_current] > 0;
        bool value_left =
            (x > 0 ? volume[offset_left] : outside_voxels_value) > 0;
        bool value_back =
            (y > 0 ? volume[offset_back] : outside_voxels_value) > 0;
        bool value_up =
            (z > 0 ? volume[offset_up] : outside_voxels_value) > 0;

        int64 &region_current = regions[offset_current];
        int64 region_left = x > 0 ? regions[offset_left] : outside_region;
        int64 region_back = y > 0 ? regions[offset_back] : outside_region;
        int64 region_up = z > 0 ? regions[offset_up] : outside_region;

        if (value_current == value_left && value_current == value_up) {
          union_set.Merge(region_left, region_up);
        }

        if (value_current == value_left && value_current == value_back) {
          union_set.Merge(region_left, region_back);
        }

        if (value_current == value_up && value_current == value_back) {
          union_set.Merge(region_up, region_back);
        }

        constexpr int64 invalid_candidate = std::numeric_limits<int64>::max();
        std::array<int64, 3> candidate_region_ids = {
            invalid_candidate, invalid_candidate, invalid_candidate};
        if (value_current == value_left) {
          candidate_region_ids[0] = region_left;
        }
        if (value_current == value_back) {
          candidate_region_ids[1] = region_back;
        }
        if (value_current == value_up) {
          candidate_region_ids[2] = region_up;
        }

        int64 candidate_region_id = *std::min_element(
            candidate_region_ids.begin(), candidate_region_ids.end());
        region_current = candidate_region_id != invalid_candidate
                         ? candidate_region_id
                         : union_set.NewRegion();
      }
    }
  }

  int64 size = size_z * size_y * size_x;
  for (int64 *v = regions; v < regions + size; v++) {
    *v = union_set.FindRoot(*v);
  }
}

template<class T>
void FillInsideVoxels(std::tuple<int, int, int> grid_size, T *voxel_grid) {
  auto[size_z, size_y, size_x] = grid_size;
  int64 size = size_z * size_y * size_x;
  std::vector<int64> regions(size);
  ComputeConnectedComponents(grid_size, voxel_grid, (T) 0, &regions[0]);
  for (int64 index = 0; index < size; index++) {
    if (regions[index] > 0) {
      voxel_grid[index] = 1;
    }
  }
}
}  // namespace

torch::Tensor fill_inside_voxels_cpu(const torch::Tensor grid) {
  if (!grid.device().is_cpu()) {
    throw pybind11::value_error("Only CPU tensors are supported currently");
  }

  const auto &shape = grid.sizes();
  if (shape.size() != 4) {
    throw pybind11::value_error("Expecting rank 4 tensor");
  }

  torch::Tensor output_grid = grid.clone(c10::MemoryFormat::Contiguous).cpu();

  int num_grids = shape[0];
  auto shape_tuple = std::make_tuple(shape[1], shape[2], shape[3]);
  int64 num_elements_in_grid = shape[1] * shape[2] * shape[3];
  AT_DISPATCH_ALL_TYPES(grid.scalar_type(), "FillInsideVoxels", ([&] {
    auto *data = (scalar_t *) output_grid.data_ptr();
    at::parallel_for(0, num_grids, 1, [&](int64 begin, int64 end) {
      for (int64 index = begin; index < end; index++) {
        FillInsideVoxels(shape_tuple, data + num_elements_in_grid * index);
      }
    });
  }));

  return output_grid;
}
