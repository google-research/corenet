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

// Renders a voxel grid
#version 430

layout(points) in;
layout(triangle_strip, max_vertices = 24) out;

// The voxel grid resolution: (depth, height, width)
// Each voxel in the grid is assumed to be a cube of unit size and the grid
// is assumed to start at the origin.
uniform ivec3 grid_resolution;

// Transforms the voxel grid into world space
uniform mat4 voxel_to_view_matrix;

// The view projection transformation
uniform mat4 view_projection_matrix;

// The voxel grid, expected to have
// grid_resolution.x * grid_resolution.y * grid_resolution.z entries.
// The value in each entry indexes the "voxel_colors" buffer.
layout(binding=0) buffer voxel_grid { int voxel_grid_buffer[]; };

// The voxel colors
layout(binding=1) buffer voxel_colors { float voxel_colors_buffer[]; };

struct Material
{
    vec4 ambient;
    vec4 diffuse_and_texture;
    vec4 specular_shininess;
};

out layout(location = 0) vec3 position;
out layout(location = 1) vec3 normal;
out layout(location = 2) vec2 texcoord;
out layout(location = 4) flat Material material;


void emit_quad(vec3 center, vec3 dy, vec3 dx) {
    center = (voxel_to_view_matrix * vec4(center, 1)).xyz;
    dx = (voxel_to_view_matrix * vec4(dx, 0)).xyz;
    dy = (voxel_to_view_matrix * vec4(dy, 0)).xyz;

    normal = normalize(cross(dx, dy));

    vec3 vertices[] = {(dx - dy), (-dx - dy), (dx + dy), (-dx + dy)};
    for (int i = 0; i < 4; i++) {
        position = (center + vertices[i]);
        gl_Position = view_projection_matrix * vec4(position, 1);
        EmitVertex();
    }
    EndPrimitive();
}

in int gl_PrimitiveIDIn;
void main() {
    int label = voxel_grid_buffer[gl_PrimitiveIDIn];
    vec3 color = vec3(voxel_colors_buffer[label * 3 + 0],
    voxel_colors_buffer[label * 3 + 1],
    voxel_colors_buffer[label * 3 + 2]);
    if(color.x < 0 || color.y < 0 || color.z < 0) {
        return;
    }

    texcoord = vec2(0);
    material.ambient = vec4(0, 0, 0, 1);
    material.diffuse_and_texture = vec4(color, -1);
    material.specular_shininess = vec4(0, 0, 0, 1);

    int x = gl_PrimitiveIDIn % grid_resolution.z;
    int y = (gl_PrimitiveIDIn / grid_resolution.z) % grid_resolution.y;
    int z = (gl_PrimitiveIDIn / grid_resolution.z) / grid_resolution.y;
    vec3 position = vec3(x, y, z);

    vec3 center = position + 0.5;

    vec3 dx = vec3(1, 0, 0) / 2;
    vec3 dy = vec3(0, 1, 0) / 2;
    vec3 dz = vec3(0, 0, 1) / 2;

    emit_quad(center + dx, dy, dz);
    emit_quad(center - dx, dz, dy);
    emit_quad(center + dy, dz, dx);
    emit_quad(center - dy, dx, dz);
    emit_quad(center + dz, dx, dy);
    emit_quad(center - dz, dy, dx);
}
