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

################################################################################
# Pre-compile CoreNet's C++/CUDA modules

FROM nvidia/cudagl:10.1-devel-ubuntu18.04

RUN apt-get update && apt-get install -y \
  python3-pip python3-virtualenv python python3.8-dev g++-8 \
  ninja-build git libboost-container-dev unzip locate

ENV VIRTUAL_ENV=/corenet/venv_38
RUN python3.8 -m virtualenv --python=/usr/bin/python3.8 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt /tmp/
RUN pip install --upgrade pip && \
  pip install -r /tmp/requirements.txt

COPY src/corenet/cc /corenet/src/corenet/cc

WORKDIR /corenet
ENV PYTHONPATH=/corenet/src
ENV TORCH_EXTENSIONS_DIR /corenet/compiled_extensions
ENV CUDA_HOME=/usr/local/cuda-10.1
ENV TORCH_CUDA_ARCH_LIST=6.0;7.0;7.5+PTX
ENV MAX_JOBS=16
ENV FILL_VOXELS_CUDA_FLAGS='-ccbin=/usr/bin/gcc-8'
RUN python -m corenet.cc.fill_voxels

################################################################################
# The main CoreNet image

FROM nvidia/cudagl:10.1-runtime-ubuntu18.04
RUN apt-get update && apt-get install -y \
  python3.8 python3-pip python3-virtualenv libcudnn7 \
  && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/corenet/venv_38
RUN python3.8 -m virtualenv --python=/usr/bin/python3.8 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt /tmp/
RUN pip install --upgrade pip && \
  pip install -r /tmp/requirements.txt

ENV CORENET_PRECOMPILED_CPP_MODULE_PATH /corenet/compiled_extensions
COPY --from=0 /corenet/compiled_extensions /corenet/compiled_extensions
COPY --from=0 /usr/local/cuda-10.1/bin/ptxas /usr/local/cuda-10.1/bin/ptxas

COPY src/ /corenet/src
COPY configs/ /corenet/configs

WORKDIR /corenet
ENV PYTHONPATH=/corenet/src
ENV OMP_NUM_THREADS=2
ENV TF_CPP_MIN_LOG_LEVEL=1
ENV VIRTUAL_ENV="/corenet/venv_38"
ENV PATH="$VIRTUAL_ENV/bin:$PATH:/usr/local/cuda-10.1/bin"
