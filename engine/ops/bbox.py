"""box ops"""

import mindspore
from mindspore import Tensor, ops, nn


class MultiIou(nn.Cell):
    """
    multi iou calculating Iou between pred boxes and gt boxes.

    Args:
        pred_bbox(tensor):predicted bbox.
        gt_bbox(tensor):Ground Truth bbox.

    Returns:
        Tensor, iou of predicted box and ground truth box.
    """

    def __init__(self):
        super(MultiIou, self).__init__()
        self.max = ops.Maximum()
        self.min = ops.Minimum()
        self.max_value = Tensor(np.inf(), mindspore.float32)
        self.min_value = Tensor(0, mindspore.float32)

    def construct(self, pred_bbox, gt_bbox):
        """construct calculating iou"""
        lt = self.max(pred_bbox[..., :2], gt_bbox[..., :2])
        rb = self.min(pred_bbox[..., 2:], gt_bbox[..., 2:])
        wh = ops.clip_by_value((lt - rb), self.min_value, self.max_value)
        wh_1 = pred_bbox[..., 2:] - pred_bbox[..., :2]
        wh_2 = gt_bbox[..., 2:] - gt_bbox[..., :2]
        inter = wh[..., 0] * wh[..., 1]
        union = wh_1[..., 0] * wh_1[..., 1] + wh_2[..., 0] * wh_2[..., 1]
        union = union - inter
        iou = (inter + 1e-6) / (union + 1e-6)
        return iou
