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

layout(points) in;
layout(triangle_strip, max_vertices=12) out;

uniform mat4 view_projection_matrix;

out layout(location = 0) vec3 position;
out layout(location = 1) float shape_index;

layout(binding=0) buffer buffer_0 { float mesh[]; };
layout(binding=1) buffer buffer_1 { int in_shape_index[]; };
layout(binding=2) buffer buffer_2 { mat4 view_to_voxel_matrices[]; };

in int gl_PrimitiveIDIn;

vec3 get_position(int i, mat4 view2vox) {
    int o = gl_PrimitiveIDIn * 9 + i * 3;
    vec3 position = vec3(mesh[o + 0], mesh[o + 1], mesh[o + 2]);
    return (view2vox * vec4(position, 1)).xyz;
}

void main() {
    int shape_index_int = in_shape_index[gl_PrimitiveIDIn];
    shape_index = shape_index_int;
    mat4 view2vox = transpose(view_to_voxel_matrices[shape_index_int]);
    vec3 v0 = get_position(0, view2vox);
    vec3 v1 = get_position(1, view2vox);
    vec3 v2 = get_position(2, view2vox);
    vec3 normal = normalize(cross(normalize(v1 - v0), normalize(v2 - v0)));

    vec3 positions[3] = { v0, v1, v2 };
    for (int i = 0; i < 3; i++) {
        position = positions[i];
        gl_Position = view_projection_matrix * vec4(positions[i], 1);

        // Make sure that dz/dx < 1 and dz/dy < 1 in screen space (z points inward)
        // in order to avoid holes in the grid.
        vec3 a = abs(normal);
        if (a.x > a.y && a.x > a.z) gl_Position = gl_Position.yzxw;
        if (a.y > a.x && a.y > a.z) gl_Position = gl_Position.zxyw;

        EmitVertex();
    }
    EndPrimitive();
}

