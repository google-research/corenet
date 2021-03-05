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

"""Manages model checkpoints."""

import dataclasses
import logging
from typing import List
from typing import Optional

import re

from corenet import file_system as fs

log = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class _CheckPoint:
  path: str
  step: int


class CheckpointReader:
  _PREFIX = "state_"
  _SUFFIX = ".cpt"

  def __init__(self, cpt_dir: str, refresh: bool = True):
    """Initializes the checkpoint reader.

    Args:
      cpt_dir: The checkpoint directory
      refresh: Whether to refresh the checkpoints from disk.
    """
    cpt_dir = fs.normpath(fs.abspath(cpt_dir))
    self.pers_cpt_dir = fs.join(cpt_dir, "persistent")
    self.tmp_cpt_dir = fs.join(cpt_dir, "temp")
    self.tmp_cpts = []
    self.pers_cpts = []
    if refresh:
      self.refresh()

  def refresh(self):
    """Refreshes the checkpoints from disk."""
    self.tmp_cpts = CheckpointReader._get_checkpoints(self.tmp_cpt_dir)
    self.pers_cpts = CheckpointReader._get_checkpoints(self.pers_cpt_dir)

  def has_checkpoints(self):
    """Returns true if any checkpoints exist."""
    return bool(self.tmp_cpts or self.pers_cpts)

  def read_last_checkpoint(self, force_persistent=False) -> Optional[bytes]:
    """Reads the last checkpoint.

    Args:
      force_persistent: If true, only looks at the persistent checkpoints.

    Returns:
      The contents of the last checkpoint
    """
    cpts = self.pers_cpts.copy()
    if not force_persistent:
      cpts += self.tmp_cpts
    if not cpts:
      return None
    last_cpt = sorted(cpts, key=lambda v: v.step)[-1]
    return fs.read_bytes(last_cpt.path)

  @classmethod
  def _get_checkpoints(cls, cpt_dir: str) -> List[_CheckPoint]:
    result = fs.glob_pattern(fs.join(cpt_dir, f"{cls._PREFIX}*{cls._SUFFIX}"))
    regex = rf"^{cls._PREFIX}(\d+){cls._SUFFIX}$"
    result = [(path, re.match(regex, fs.basename(path))) for path in result]
    result = [_CheckPoint(path, int(m.group(1))) for path, m in result if m]
    result = sorted(result, key=lambda v: v.step)
    return result


class CheckpointManager(CheckpointReader):

  def __init__(self, cpt_dir: str, num_temp_states_to_keep=5,
               refresh: bool = True):
    """Initializes the checkpoint manager.

    Args:
      cpt_dir: The checkpoint directory.
      num_temp_states_to_keep: How many recent temporary states to keep.
      refresh: Whether to refresh the checkpoints from disk.
    """
    super().__init__(cpt_dir, refresh=False)
    self.num_temp_states_to_keep = num_temp_states_to_keep
    fs.make_dirs(self.tmp_cpt_dir)
    fs.make_dirs(self.pers_cpt_dir)
    if refresh:
      self.refresh()

  def cleanup_temporary_checkpoints(self):
    """Deletes all but the last few temporary checkpoints."""
    self.tmp_cpts = sorted(self.tmp_cpts, key=lambda v: v.step)
    cpts_to_delete = self.tmp_cpts[:-self.num_temp_states_to_keep]
    self.tmp_cpts = self.tmp_cpts[-self.num_temp_states_to_keep:]
    for cpt in cpts_to_delete:
      try:
        fs.unlink_file(cpt.path)
      except OSError:
        log.exception(f"Error deleting checkpoint {cpt.path}")

  def save_state(self, state: bytes, step: int, persistent=False):
    if persistent:
      save_dir = self.pers_cpt_dir
      cpt_collection = self.pers_cpts
    else:
      save_dir = self.tmp_cpt_dir
      cpt_collection = self.tmp_cpts

    # Save in two stages to avoid corruption
    CM = CheckpointManager
    temp_path = fs.join(save_dir, f"temporary_state.{step:09}{CM._SUFFIX}")
    fs.write_bytes(temp_path, state)
    save_path = fs.join(save_dir, f"{CM._PREFIX}{step:09}{CM._SUFFIX}")
    fs.rename_file(temp_path, save_path)
    cpt_collection.append(_CheckPoint(save_path, step))

    self.cleanup_temporary_checkpoints()
