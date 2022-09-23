"""DGCNN segmentation training script."""

import os
import mindspore.nn as nn
from mindspore import context,FixedLossScaleManager
from mindspore.communication.management import init,get_rank,get_group_size
from mindspore.train.callback import ModelCheckpoint,CheckpointConfig,LossMonitor,TimeMonitor
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.common import set_seed
import mindspore.dataset as ds
from src.model_utils.config import config
from dataset.S3DIS_V2 import S3DISDataset
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id,get_device_num
from models.dgcnn import DGCNN_seg
from utils.Lossfunction import CrossEntropySmooth_SEG
from engine.callback.monitor import ValAccMonitor

set_seed(1)

@moxing_wrapper()
def train_net():
    """
    Mindspore training DGCNN SEG
    """
    if config.device_target == 'Ascend':
        device_id = int(os.getenv('DEVICE_ID'), 0)
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False,
                            device_id=device_id)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
    print(config.run_distribute)

    if config.run_distribute:
        init()
        if config.device_target=="Ascend":
            rank_id=get_device_id()
            rank_size=get_device_num()
        else:
            rank_id=get_rank()
            rank_size=get_group_size()
        parallel_mode=ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          device_num=rank_size,
                                          gradients_mean=False)
    else:
        rank_id=0
        rank_size=1

    # Data Pipeline.
    data_train = S3DISDataset(split="train",num_points=config.num_point,test_area=config.test_area)
    trainds = ds.GeneratorDataset(data_train, ["data", "label"], shuffle=True)
    trainds = trainds.batch(config.batch_size, drop_remainder=True)

    train_data_size = trainds.get_dataset_size()

    data_val = S3DISDataset(split='test',num_points=config.num_point,test_area=config.test_area)
    valds = ds.GeneratorDataset(data_val, ["data", "label"], shuffle=False)
    valds = valds.batch(config.val_batch)
    print("train dataset length is:", train_data_size)

    #Create model.
    network=DGCNN_seg(config,config.k)

    #Define loss function.
    loss=CrossEntropySmooth_SEG(num_classes=13,reduction="mean",sparse=True)

    # Set learning rate scheduler.
    if config.lr_decay_mode == "cosine_decay_lr":
        lr = nn.cosine_decay_lr(min_lr=config.min_lr,
                                max_lr=config.max_lr,
                                total_step=config.epoch_size * train_data_size,
                                step_per_epoch=train_data_size,
                                decay_epoch=config.decay_epoch)
    elif config.lr_decay_mode == "piecewise_constant_lr":
        lr = nn.piecewise_constant_lr(config.milestone, config.learning_rates)

    #Define optimizer.
    optimizer=nn.Adam(network.trainable_params(),lr,config.momentum)

    #Define metrics.
    metrics={"Accuracy": nn.Accuracy(eval_type="classification")}

    #Init the model.
    network.set_train()
    if config.device_target=="CPU" and config.enable_fp16_gpu:
        model=Model(network,loss_fn=loss,optimizer=optimizer, amp_level='O2', metrics=metrics)
    else:
        model=Model(network,loss_fn=loss,optimizer=optimizer, metrics=metrics)

    #Set the checkpoint config for the network.
    ckpt_config=CheckpointConfig(save_checkpoint_steps=train_data_size,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    ckpt_callback=ModelCheckpoint(prefix="DGCNN_seg",
                                  directory=config.ckpt_save_dir,
                                  config=ckpt_config)

    #Begin to train.
    print("============ Starting Training ================")
    model.train(config.epoch_size,
                trainds,
                callbacks=[ValAccMonitor(model,valds,config.epoch_size)],
                dataset_sink_mode=config.dataset_sink_mode,)
    print("============ End Training ====================")

if __name__=="__main__":
    train_net()