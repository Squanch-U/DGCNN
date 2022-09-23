from unittest.mock import patch
import numpy as np
import pytest
import mindspore as ms
import mindspore.ops.operations as op
from mindspore import nn,set_context,PYNATIVE_MODE

set_context(mode=PYNATIVE_MODE)

class CustomNet(nn.Cell):
    """Simple net for test."""
    def __init__(self):
        super(CustomNet,self).__init__()
        self.fc1=nn.Dense(10,10)
        self.fc2=nn.Dense(10,10)
        self.fc3=nn.Dense(10,10)
        self.fc4=nn.Dense(10,10)

    def construct(self, inputs):
        out=self.fc1(inputs)
        out=self.fc2(out)
        out=self.fc3(out)
        out=self.fc4(out)
        return out

# @pytest.mark.level0
# @pytest.mark.platform_arm_ascend_training
# @pytest.mark.platform_x86_ascend_training
# @pytest.mark.platform_x86_gpu_training
# @pytest.mark.env_onecard
# def test_rank_pixels():
#     """Test on rank_pixels."""
#     saliency=np.array([[4.,3.,1.],[5.,9.,1.]])
#     descending_target=np.array([[0,1,2],[1,0,2]])
#     ascending_target=np.array([[2,1,0],[1,2,0]])
#     descending_rank=rank_pixels(saliency)
#     ascending_rank=rank_pixels(saliency,descending=False)
#     assert (descending_rank-descending_target).any()==0
#     assert (ascending_rank-ascending_target).any()==0

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_retrieve_layer_by_name():
    """Test on rank_pixels."""
    model=CustomNet()
    target_layer_name="fc3"
    target_layer=retrieve_layer_by_name(model,target_layer_name)




