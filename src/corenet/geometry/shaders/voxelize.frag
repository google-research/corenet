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

#version 430

layout(origin_upper_left) in vec4 gl_FragCoord;

in layout(location = 0) vec3 position;
in layout(location = 1) float shape_index;

out vec4 output_color;

layout(binding = 5) buffer voxel_grid_buffer { float voxel_grid[]; };
uniform ivec4 voxel_grid_shape;// #W, H, D, N
uniform int virtual_voxel_side;


void main() {
  output_color = vec4(1.0);
  // The position in voxel space, equal to the fractional voxel index
  vec3 p = position.xyz;
  int w = voxel_grid_shape.x;
  int h = voxel_grid_shape.y;
  int d = voxel_grid_shape.z;
  if (p.x < 0 || p.y < 0 || p.z < 0 || p.x >= w || p.y >= h || p.z >= d) {
    return;
  }

  if (virtual_voxel_side <= 0) {
    // Voxel index in the real voxel grid
    ivec3 c = ivec3(floor(position.xyz));
    int a = int(round(shape_index));
    a = ((a * d + c.z) * h + c.y) * w + c.x;
    voxel_grid[a] = 1;
  } else {
    // Voxel index in the virtual voxel grid
    ivec3 v = ivec3(floor(position.xyz * float(virtual_voxel_side)));
    v += virtual_voxel_side / 2;
    ivec3 c = v / virtual_voxel_side;
    bvec3 r = equal(v % virtual_voxel_side, vec3(virtual_voxel_side - 1));
    c = 2 * c + ivec3(r);

    int a = int(round(shape_index));
    a = ((a * (2 * d + 1) + c.z) * (2 * h  + 1) + c.y) * (2 * w + 1) + c.x;
    voxel_grid[a] = 1;
  }
}

