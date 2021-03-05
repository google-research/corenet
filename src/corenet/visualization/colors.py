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

"""Defines a color palette for visualization."""

import numpy as np

_DEFAULT_INTEGER_COLOR_PALETTE = (
    (-255, -255, -255),
    (120, 120, 120), (180, 120, 120), (6, 230, 230),
    (80, 50, 50), (4, 200, 3), (120, 120, 80), (140, 140, 140),
    (204, 5, 255), (230, 230, 230), (4, 250, 7), (224, 5, 255),
    (235, 255, 7), (150, 5, 61), (120, 120, 70), (8, 255, 51),
    (255, 6, 82),
)  # pyformat: disable

DEFAULT_COLOR_PALETTE = np.array(_DEFAULT_INTEGER_COLOR_PALETTE,
                                 np.float32) / 255.0
