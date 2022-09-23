"""KNN module"""
from mindspore import ops
"""KNN module"""


def KNN(x, k):
    """
           KNN Module.

           The input data is x shape(B,C,N)
           where B is the batch size ,C is the dimension of the transform matrix
           and N is the number of points.

           :param x: input data
           :param k: k-NearestNeighbor Parameter
           :return: Tensor shape(B,N,K)
       """
    inner = -2 * ops.matmul(x.transpose(0, 2, 1), x)
    xx = (x ** 2).sum(axis=1, keepdims=True)
    pairwise_distance = -xx - inner - xx.transpose(0, 2, 1)
    topk = ops.TopK(sorted=True)
    _, idx = topk(pairwise_distance, k)
    return idx
