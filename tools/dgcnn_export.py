"""
This module is used to export ckpt file to mindir/air model
"""

import os
import numpy as np
from mindspore import dtype as  mstype
from mindspore import context,Tensor,export
from mindspore.train.serialization import load_checkpoint,load_param_into_net
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from models.dgcnn import DGCNN_cls

device_id=int(os.getenv("DEVICE_ID"))
context.set_context(mode=context.GRAPH_MODE,device_target=config.device_target,\
                    save_graphs=False,device_id=device_id)

@moxing_wrapper()
def export_model(ckpt_path):
    """
    Args:
        ckpt_path: cpkt_path: The file path location eg. /home/xxx.ckpt

    Returns:
        None
    """
    network=DGCNN_cls(config,output_channels=config.num_classes)
    network.set_train(False)
    param_dict=load_checkpoint(ckpt_path)
    load_param_into_net(network,param_dict)
    #image shape is the input of the network
    image_shape=[config.infer_batch_size,config.in_channels]+config.roi_size
    window_image=Tensor(np.zeors(image_shape),mstype.float32)
    export(network,window_image,file_name=config.file_name,file_format=config.file_format)

if __name__=="__main__":
    export_model(ckpt_path=config.cpkt_file)