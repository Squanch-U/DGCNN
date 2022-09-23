"""Dgcnn classification eval script"""

import os
import mindspore.nn as nn
from mindspore import context, FixedLossScaleManager
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_param_into_net, load_checkpoint
from mindspore.common import set_seed
import mindspore.dataset as ds
from src.model_utils.config import config
from dataset.ModelNet40v1 import ModelNet40Dataset
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id,get_device_num
from models.dgcnn import DGCNN_cls
from utils.Lossfunction import CrossEntropySmooth_CLS
from engine.callback.monitor import ValAccMonitor

set_seed(1)

@moxing_wrapper()
def eval_net():
    """
    Dgcnn eval net defintion
    """


    if config.device_target == 'Ascend':
        device_id = int(os.getenv('DEVICE_ID'), 0)
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False, \
                            device_id=device_id)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
    print(config.run_distribute)

    if config.run_distribute:
        init()
        if config.device_target == "Ascend":
            rank_id = get_device_id()
            rank_size = get_device_num()
        else:
            rank_id = get_rank()
            rank_size = get_group_size()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          device_num=rank_size,
                                          gradients_mean=False)
    else:
        rank_id = 0
        rank_size = 1

    #Data Pipeline.
    data_val = ModelNet40Dataset(config.val_dir, split='val', use_norm=False, num_points=1024)
    valds = ds.GeneratorDataset(data_val, ["data", "label"], shuffle=False)
    valds = valds.batch(config.val_batch)

    eval_data_size=valds.get_dataset_size()
    print("eval dataset length is: ",eval_data_size)

    #Create model
    network=DGCNN_cls(config,num_classes=config.num_classes)

    #Load param
    param_dict=load_checkpoint(config.cls_ckpt_file)
    load_param_into_net(network,param_dict)

    # Define loss function.
    loss = CrossEntropySmooth_CLS()

    #Define metrics
    metrics={"Accuracy": nn.Accuracy(eval_type="classification")}

    # Init the model.
    network.set_train()
    if config.device_target == "CPU" and config.enable_fp16_gpu:
        model = Model(network, loss_fn=loss, amp_level='O2', metrics=metrics)
    else:
        model = Model(network, loss_fn=loss, metrics=metrics)

    #Begin to eval
    result=model.eval(valds,dataset_sink_mode=True)
    print(result)

if __name__=="__main__":
    eval_net()
