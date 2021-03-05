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

"""Helper routines for distributed training/evaluation."""
import dataclasses
import pickle
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar
from urllib.parse import urlparse

import numpy as np
import os
import torch as t
import torch.distributed.rpc
import torch.utils.data


@dataclasses.dataclass(frozen=True)
class DistributedInfo:
  global_rank: int
  local_rank: int
  group_rank: int
  global_world_size: int
  local_world_size: int
  master_addr: str
  master_port: int


_distributed_info: Optional[DistributedInfo] = None
_local_group = None
_rendezvous_root_stores = {}
_gloo_group = None


# noinspection PyUnusedLocal
def _rendezvous_handler(url_str: str, timeout=t.distributed.default_pg_timeout,
                        **kwargs):
  """Allows to initialize both RPC and process_group on the same port."""

  def _error(msg):
    raise ValueError("Failed to construct corenet:// rendezvous: " + msg)

  url = urlparse(url_str)
  if not url.port:
    _error("port number missing")
  params = dict((p[0], p[1])
                for pair in url.query.split("&")
                if len(p := pair.split("=", 1)) == 2)
  if "rank" not in params:
    _error("rank parameter missing")
  if "world_size" not in params:
    _error("world_size parameter missing")
  if "prefix" not in params:
    _error("prefix parameter missing")

  rank = int(params["rank"])
  world_size = int(params["world_size"])
  prefix = params["prefix"]
  root_key = f"{url.hostname}:{url.port}"
  if root_key not in _rendezvous_root_stores:
    _rendezvous_root_stores[root_key] = t.distributed.TCPStore(
        url.hostname, url.port, world_size, rank == 0, timeout)

  root_store = _rendezvous_root_stores[root_key]
  store = t.distributed.PrefixStore(prefix, root_store)
  yield store, rank, world_size

  # If this configuration is invalidated, there is nothing we can do about it
  raise RuntimeError("Unable to perform rendezvous using corenet:// method")


t.distributed.register_rendezvous_handler("corenet", _rendezvous_handler)


def get_node_name(rank: int):
  return f"node_{rank:02}"


def shutdown():
  t.distributed.rpc.shutdown()


def init():
  # Read the distributed configuration from environment variables
  global _distributed_info
  global_world_size = int(os.environ.get("WORLD_SIZE", 1))
  global_rank = int(os.environ.get("RANK", 0))
  _distributed_info = DistributedInfo(
      global_rank=global_rank,
      local_rank=int(os.environ.get("LOCAL_RANK", global_rank)),
      group_rank=int(os.environ.get("GROUP_RANK", 0)),
      global_world_size=global_world_size,
      local_world_size=int(
          os.environ.get("LOCAL_WORLD_SIZE", global_world_size)),
      master_addr=os.environ.get("MASTER_ADDR", "127.0.0.1"),
      master_port=int(os.environ.get("MASTER_PORT", 6778)))

  i = _distributed_info
  valid_config = (
      i.global_world_size >= i.local_world_size > 0 and
      i.global_world_size % i.local_world_size == 0 and
      i.local_rank < i.local_world_size and i.global_rank < i.global_world_size
      and i.group_rank < i.global_world_size / i.local_world_size)
  if not valid_config:
    raise ValueError(f"Invalid distributed configuration detected: {i}")

  # Initialize torch.distributed
  init_method = f"corenet://{i.master_addr}:{i.master_port}?prefix=group"
  t.distributed.init_process_group("nccl", world_size=i.global_world_size,
                                   rank=i.global_rank, init_method=init_method)
  global _gloo_group
  _gloo_group = t.distributed.new_group(backend="gloo")

  # Initialize torch.distributed.rpc
  backend_options = t.distributed.rpc.TensorPipeRpcBackendOptions(
      init_method=f"corenet://{i.master_addr}:{i.master_port}?prefix=rpc")
  t.distributed.rpc.init_rpc(
      get_node_name(i.global_rank), rank=i.global_rank,
      world_size=i.global_world_size, rpc_backend_options=backend_options)

  # Create a group with all workers on the current node
  global _local_group
  local_worker_ranks = list(range(i.group_rank * i.local_world_size,
                                  (i.group_rank + 1) * i.local_world_size))
  _local_group = t.distributed.new_group(local_worker_ranks)


def info() -> DistributedInfo:
  """Returns the distributed configuration, reading from environment variables."""
  if not _distributed_info:
    raise ValueError("Please call init() first!")
  return _distributed_info


def local_nccl_group():
  """Returns a group name for all workers on the current node."""
  if not _local_group:
    raise ValueError("Please call init() first!")
  return _local_group


def global_gloo_group():
  if not _gloo_group:
    raise ValueError("Please call init() first!")
  return _gloo_group


def get_worker_range(total: int) -> Tuple[int, int]:
  """Computes a processing range for a worker from a given total."""
  dist_info = info()
  start = (dist_info.global_rank * total) // dist_info.global_world_size
  end = ((dist_info.global_rank + 1) * total) // dist_info.global_world_size
  return start, end


T = TypeVar("T")


def gather(o: T, dst: int) -> Optional[List[T]]:
  """Gather for arbitrary objects."""
  serialized_tensor = t.from_numpy(
      np.array(np.frombuffer(pickle.dumps(o), dtype=np.uint8)))

  shape = t.scalar_tensor(serialized_tensor.shape[0], dtype=t.int64,
                          device="cpu")
  len_list = [t.scalar_tensor(0, dtype=t.int64, device="cpu")
              for _ in range(info().global_world_size)]
  t.distributed.all_gather(len_list, shape, group=global_gloo_group())
  max_len = int(max(len_list))
  serialized_tensor = t.nn.functional.pad(
      serialized_tensor, [0, max_len - serialized_tensor.shape[0]])

  if info().global_rank == dst:
    result = [
        t.zeros([max_len], dtype=t.uint8, device="cpu") for _ in len_list]
  else:
    result = None

  t.distributed.gather(
      serialized_tensor, gather_list=result, dst=dst,
      group=global_gloo_group())

  if info().global_rank == dst:
    result = [v[:s] for v, s in zip(result, len_list)]
    result = [pickle.loads(bytes(v.numpy())) for v in result]
    return result
  else:
    return None


class DistributedSampler(t.utils.data.Sampler):
  """A distributed data sampler."""

  def __init__(self, dataset: t.utils.data.Dataset, global_rank: int,
               global_world_size: int, pad_data: bool):
    super().__init__(dataset)
    if pad_data:
      total_size = (len(dataset) + global_world_size - 1) // global_world_size
      total_size *= global_world_size
    else:
      total_size = len(dataset)

    g = t.Generator()
    # Shuffle data among workers in a stable way.
    g.manual_seed(0x1234)
    indices = t.randperm(len(dataset), generator=g)
    indices = t.constant_pad_nd(indices, [0, total_size - indices.shape[0]])

    start = global_rank * total_size // global_world_size
    end = (global_rank + 1) * total_size // global_world_size
    self.indices = indices[start: end]

  def __iter__(self):
    return iter(self.indices)

  def __len__(self):
    return self.indices.shape[0]
