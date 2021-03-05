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

"""Voxel evaluation metrics."""
import dataclasses as d
import math

import torch as t

from corenet import misc_util


@d.dataclass
class TfpnValues(misc_util.TensorContainerMixin):
  """True/false positives/negatives. All shapes are float32[num_classes]."""
  tp: t.Tensor
  tn: t.Tensor
  fp: t.Tensor
  fn: t.Tensor


def confusion_matrix(predicted: t.Tensor, gt: t.Tensor,
                     num_classes: int) -> t.Tensor:
  """Computes the confusion metrics between the predicted and the GT labels.

  Args:
    predicted: Predicted labels, int[...]
    gt: GT labels, int[...]
    num_classes: The number of classes

  Returns:
    The confusion matrix, int32[num_classes, num_classes]. The first dimension
    corresponds to GT, the second -- to predicted.
  """
  valid_dtypes = {t.int32, t.int64}
  assert (predicted.shape == gt.shape and predicted.dtype in valid_dtypes and
          gt.dtype in valid_dtypes)
  predicted = predicted.reshape([-1]).to(t.int64)
  gt = gt.reshape([-1]).to(t.int64)
  assert predicted.max().item() < num_classes and gt.max().item() < num_classes

  index = (gt * num_classes + predicted).reshape([-1])
  values = t.ones_like(index, dtype=t.int32)
  result = gt.new_zeros(num_classes ** 2, dtype=t.int32)

  result = result.scatter_add(0, index, values).reshape([num_classes] * 2)
  return result


def compute_tfpn(matrix: t.Tensor) -> TfpnValues:
  """Computes the true/false positive/negatives from a confusion matrix.

  Args:
    matrix: The confusion matrix, float64[num_classes, num_classes].
      The X dimension is the predicted class, the Y -- the GT class.

  Returns:
    tp: The true positive counts, float64[num_classes]
    tn: The true negative counts, float64[num_classes]
    fp: The false positive counts, float64[num_classes]
    fn: The false negative counts, float64[num_classes]
  """
  num_classes = matrix.shape[0]
  assert matrix.shape == (num_classes,) * 2

  # Example:
  # Class=2, true/false positive/negative counts in the confusion matrix
  # Axis 0 (y) is GT, axis 1 (x) is predicted.
  #      PRED
  #   tn tn fp tn
  # G tn tn fp tn
  # T fn fn tp fn
  #   tn tn fp tn

  tp = matrix.diagonal()

  sum_collapse_gt = matrix.sum(dim=0)
  fp = sum_collapse_gt - tp

  sum_collapse_pred = matrix.sum(dim=1)
  fn = sum_collapse_pred - tp

  total_num_labels = matrix.sum().expand([num_classes])
  tn = total_num_labels - tp - fp - fn

  return TfpnValues(tp, tn, fp, fn)


def compute_tfpn_fg(matrix: t.Tensor) -> TfpnValues:
  """Computes 2-class foreground/background TFPNs from a confusion matrix."""
  tp = matrix[1:, 1:].sum()
  tn = matrix[0, 0]
  fp = matrix[0, 1:].sum()
  fn = matrix[1:, 0].sum()

  return TfpnValues(tp, tn, fp, fn)


@d.dataclass
class VoxelMetrics(misc_util.TensorContainerMixin):
  """Voxel metrics, shapes are float32[num_classes]"""
  iou: t.Tensor
  precision: t.Tensor
  recall: t.Tensor


def nan_tp_div(tp: t.Tensor, y: t.Tensor):
  """Returns NaN if the class has no ground-truth (i.e. tp == 0)."""
  return t.where(tp == 0, math.nan, tp / y)


def compute_voxel_metrics(tfpn: TfpnValues) -> VoxelMetrics:
  """Computes a voxel metrics from true/false positive/negatives.

  Args:
    tfpn: The true/false positives/negatives.
  Returns: The computed metrics. Dimensionality of each metric matches the
    dimensionality of the tfpn values. For classes with no positive GT examples,
    the metrics will be NaNs.
  """
  tfpn = TfpnValues(*[v.to(t.float64) for v in d.astuple(tfpn)])
  tp, tn, fp, fn = tfpn.tp, tfpn.tn, tfpn.fp, tfpn.fn
  iou = nan_tp_div(tp, (tp + fp + fn))
  precision = nan_tp_div(tp, (tp + fp))
  recall = nan_tp_div(tp, (tp + fn))

  return VoxelMetrics(iou=iou, precision=precision, recall=recall)
