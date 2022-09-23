
"""EdgeConv module"""
from collections import OrderedDict
from models.blocks.spatial_transform import Getgraphfeature
import mindspore.ops as ops
import mindspore
import mindspore.numpy as np
import mindspore.nn as nn
from utils.knn import KNN

def Getgraphfeature(x,k=20,idx=None,dim9=False):

    batch_size=x.shape[0]
    num_points=x.shape[2]
    x=x.view(batch_size,-1,num_points)
    if idx is None:
        if dim9==False:
            idx=KNN(x,k=k)
        else:
            idx=KNN(x[:,6:],k=k)
    idx_base=np.arange(0,batch_size,dtype=mindspore.int32).view(-1,1,1)*num_points
    idx=idx+idx_base
    idx=idx.view(-1)

    _,num_dims,_=x.shape
    x=x.transpose(0,2,1)
    #print(idx)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims)

    x=np.tile(x,(1,1,k,1))
    #x = np.repeat(x, k, axis=2)

    op = ops.Concat(3)
    feature = op((feature - x, x)).transpose(0, 3, 1, 2)  # .permute(0,3,1,2) #2 6 1024 20
    return feature


def conv_bn_block(input,output,kernel_size):
    """

    Args:
        input:
        output:
        kernel_size:

    Returns:

    """
    return nn.SequentialCell(nn.Conv2d(input,output,kernel_size),
                             nn.BatchNorm2d(output),
                             nn.LeakyReLU())

class EdgeConv(nn.Cell):

    def __init__(self,layers,dim9,K=20):
        """

        Args:
            layers:
            K:
        """
        super(EdgeConv,self).__init__()
        self.K=K
        self.layers=layers
        self.dim9=dim9
        if self.layers is None:
            self.mlp=None
        else:
            mlp_layers=OrderedDict()
            for i in range(len(self.layers)-1):
                if i==0:
                    mlp_layers["conv_bn_blcok_{}".format(i+1)]=conv_bn_block(self.layers[i],self.layers[i+1],1)
                else:
                    mlp_layers["conv_bn_blcok_{}".format(i+1)]=conv_bn_block(self.layers[i],self.layers[i+1],1)
            self.mlp=nn.SequentialCell(mlp_layers)


    def construct(self,x):
        x=Getgraphfeature(x,k=self.K,dim9=self.dim9)
        x=self.mlp(x)
        argmax=ops.ArgMaxWithValue(axis=-1)
        index,x=argmax(x)
        #x=x.max(axis=-1,keepdims=False)[0]

        return x