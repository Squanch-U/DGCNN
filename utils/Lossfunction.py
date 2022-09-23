"""Calculate cross entropy loss"""
from mindspore import Tensor
import mindspore
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.operations as P
from mindspore.nn.loss.loss import LossBase
from mindspore.ops import functional as F
import mindspore.numpy as np
__all__=["CrossEntropySmooth_SEG","CrossEntropySmooth_CLS"]

# class DGCNNLoss(nn.Cell):
#     ''' Calculate cross entropy loss, apply label smoothing if needed. '''
#     def __init__(self,num_class=2,smoothing=True):
#         super(DGCNNLoss, self).__init__()
#         self.base_loss=ops.SoftmaxCrossEntropyWithLogits()
#         self.reshape=ops.Reshape()
#         self.num_class=num_class
#         self.smoothing=smoothing
#
#     def construct(self, *inputs):
#         preds,target=inputs
#         target = np.ravel(target)
#         target=target.view(-1)
#         if self.smoothing:
#             eps=0.2
#             n_class=preds.shape[1]
#             zeroslike = ops.ZerosLike()
#             op = ops.ScatterNd()
#             #one_hot = zeroslike(preds).scatter(1, target.view(-1, 1), 1)
#             one_hot=zeroslike(preds).astype("int32")
#             one_hot = op(one_hot, target.view(-1,1), (1,1))
#
#             one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
#             log_softmax = ops.LogSoftmax(axis=1)
#             log_prb=log_softmax(preds)
#             loss = -(one_hot * log_prb).sum(axis=1).mean()
#         else:
#             loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True,reduction="mean")
#             loss =loss(preds, target)
#
#         return loss



class CrossEntropySmooth_CLS(LossBase):
    """CrossEntropy for cls"""
    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth_CLS, self).__init__()
        self.onehot = ops.OneHot()
        self.sparse = sparse
        self.on_value = ms.Tensor(1.0 - smooth_factor, ms.float32)
        self.off_value = ms.Tensor(1.0 * smooth_factor / (num_classes - 1), ms.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, label)
        return loss




class CrossEntropySmooth_SEG(LossBase):
    """CrossEntropy for seg"""
    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth_SEG, self).__init__()
        self.onehot = ops.OneHot()
        self.sparse = sparse
        self.on_value = ms.Tensor(1.0 - smooth_factor, ms.float32)
        self.off_value = ms.Tensor(1.0 * smooth_factor / (num_classes - 1), ms.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):

        logit=logit.transpose(0,2,1)
        logit=logit.view(-1,13)
        label=label.view(-1,1).squeeze()
        if self.sparse:
            label = self.onehot(label, ops.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, label)
        return loss