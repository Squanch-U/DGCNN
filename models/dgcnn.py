import mindspore.ops as ops
import mindspore.nn as nn
from models.blocks.edgeconv import EdgeConv
import mindspore.numpy as np

class DGCNN_cls(nn.Cell):
    def __init__(self, args, output_channels=40):
        super(DGCNN_cls, self).__init__()
        self.args = args
        self.k = args.k

        self.edge_conv1 = EdgeConv(layers=[6, 64], K=self.k, dim9=False)
        self.edge_conv2 = EdgeConv(layers=[64 * 2, 64], K=self.k, dim9=False)
        self.edge_conv3 = EdgeConv(layers=[64 * 2, 128], K=self.k, dim9=False)
        self.edge_conv4 = EdgeConv(layers=[128 * 2, 256], K=self.k, dim9=False)
        self.conv5 = nn.Conv1d(512, args.emb_dims, kernel_size=1)
        self.bn5 = nn.BatchNorm2d(args.emb_dims)
        self.leakyRelu = nn.LeakyReLU()
        self.linear1 = nn.Dense(args.emb_dims * 2, 512, has_bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(args.dropout)
        self.linear2 = nn.Dense(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(args.dropout)
        self.linear3 = nn.Dense(256, output_channels)
        self.maxpool = nn.MaxPool1d(1024)  # 1024
        self.avepool = nn.AvgPool1d(1024)  # 1024

    def construct(self, x):
        B, N, C = x.shape
        x=x.transpose(0,2,1)
        x1 = self.edge_conv1(x)
        x2 = self.edge_conv2(x1)
        x3 = self.edge_conv3(x2)
        x4 = self.edge_conv4(x3)

        cat = ops.Concat(1)

        x = cat((x1, x2, x3, x4))
        x = self.conv5(x)
        x = ops.ExpandDims()(x, -1)
        x = self.leakyRelu(self.bn5(x))
        squeeze = ops.Squeeze(-1)
        x = squeeze(x)

        x1 = self.maxpool(x).view(B, -1)
        x2 = self.avepool(x).view(B, -1)
        x = cat((x1, x2))

        x = self.leakyRelu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.leakyRelu(self.bn7(self.linear2(x)))
        x = self.dp2(x)
        x = self.linear3(x)

        return x


class DGCNN_seg(nn.Cell):
    def __init__(self, args, k):
        super().__init__()
        self.num_classes = args.num_classes
        self.k = k

        self.edge_conv1 = EdgeConv(layers=[18, 64, 64], K=self.k, dim9=True)
        self.edge_conv2 = EdgeConv(layers=[64 * 2, 64, 64], K=self.k, dim9=False)
        self.edge_conv3 = EdgeConv(layers=[64 * 2, 64], K=self.k, dim9=False)
        self.conv6 = nn.Conv1d(192, 1024, kernel_size=1, has_bias=False)

        self.conv7 = nn.Conv1d(1216, 512, kernel_size=1, has_bias=False)
        self.conv8 = nn.Conv1d(512, 256, kernel_size=1, has_bias=False)
        self.dp1 = nn.Dropout(0.5)
        self.conv9 = nn.Conv1d(256, self.num_classes, kernel_size=1, has_bias=False)
        self.bn6 = nn.BatchNorm2d(1024)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(256)
        self.leakyRelU = nn.LeakyReLU()

    def construct(self, x):
        B, N, C = x.shape
        x = x.transpose(0, 2, 1)
        x1 = self.edge_conv1(x)

        x2 = self.edge_conv2(x1)

        x3 = self.edge_conv3(x2)

        cat = ops.Concat(1)
        x = cat((x1, x2, x3))

        x = self.conv6(x)  # batch_size,64*3,num_points
        x = ops.ExpandDims()(x, -1)
        x = self.leakyRelU(self.bn6(x))
        squeeze = ops.Squeeze(-1)
        x = squeeze(x)
        argmax = ops.ArgMaxWithValue(axis=-1, keep_dims=True)
        index, x = argmax(x)
        x = np.tile(x, (1, 1, 4096))  # (batch_size,1024,num_points)

        x = cat((x, x1, x2, x3))  # (batch_size,1024+64*3,num_points)

        x = self.conv7(x)  # (batch_size,1024+64*3,num_points)->(batch_size,512,num_points)
        x = ops.ExpandDims()(x, -1)
        x = self.bn7(x)
        x = self.leakyRelU(ops.Squeeze(-1)(x))

        x = self.conv8(x)  # (batch_size,512,num_points)->(batch_size,256,num_points)
        x = ops.ExpandDims()(x, -1)
        x = self.bn8(x)
        x = self.leakyRelU(ops.Squeeze(-1)(x))
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size,256,num_points)->(batch_size,13,num_points)

        return x