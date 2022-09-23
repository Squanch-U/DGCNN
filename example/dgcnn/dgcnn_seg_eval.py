""" Dgcnn segmentation eval script."""
import argparse
import os
import logging
import importlib
import mindspore.numpy as np
from tqdm import tqdm
from models.dgcnn import DGCNN_seg
from mindspore import context,load_checkpoint,load_param_into_net
from mindspore.common import set_seed
import mindspore.dataset as ds
import mindspore.ops as ops
from dataset.S3DIS_V2 import S3DISDataset
from src.model_utils.config import config
import sklearn.metrics as metrics

def calculate_sem_IoU(pred_np, seg_np, visual=False):
    I_all = np.zeros(13)
    U_all = np.zeros(13)
    for sem_idx in range(seg_np.shape[0]):
        for sem in range(13):
            I = np.sum(np.logical_and(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            U = np.sum(np.logical_or(pred_np[sem_idx] == sem, seg_np[sem_idx] == sem))
            I_all[sem] += I
            U_all[sem] += U
    if visual:
        for sem in range(13):
            if U_all[sem] == 0:
                I_all[sem] = 1
                U_all[sem] = 1

    return I_all / U_all

def eval_net(args):
    print("=========staring==========")
    all_true_cls = []
    all_pred_cls = []
    all_true_seg = []
    all_pred_seg = []
    for test_area in range(1,7):
        test_area=str(test_area)
        if os.path.exists("/home/cxh/文档/seu3dlab/ms3d/dataset/data/indoor3d_sem_seg_hdf5_data_test/room_filelist.txt"):
            with open("/home/cxh/文档/seu3dlab/ms3d/dataset/data/indoor3d_sem_seg_hdf5_data_test/room_filelist.txt") as f:
                for line in f:
                    if (line[5]) == test_area:
                        break
        if (args.test_area == "all") or (test_area == args.test_area):
            test_dataset_generator=S3DISDataset(split="eval",num_points=config.num_point,test_area=config.test_area)

            test_ds = ds.GeneratorDataset(test_dataset_generator, ["data", "label"], shuffle=True)
            test_ds = test_ds.batch(batch_size=1)

             #Create model
            model=DGCNN_seg(args,args.k)


            path = os.path.join(args.pretrain_path, 'model_%s.ckpt' % test_area)
            print(path)
            param_dict = load_checkpoint(path)
            load_param_into_net(model, param_dict)
            print("sucessfully load pretrain model")
            model.set_train(False)

            test_acc = 0.0
            count = 0.0
            test_true_cls = []
            test_pred_cls = []
            test_true_seg = []
            test_pred_seg = []
            for _, data in tqdm(enumerate(test_ds.create_dict_iterator(), 0)):
                points, target = data['data'], data['label']
                seg_pred = model(points)
                seg_pred=seg_pred.transpose(0,2,1)
                argmax=ops.ArgMaxWithValue(axis=2)
                index,pred=argmax(seg_pred)
                pred_np=index
                seg_np=target

                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)

                test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls.append(pred_np.reshape(-1))

            print("1")
            test_true_cls = np.concatenate(test_true_cls)
            test_pred_cls = np.concatenate(test_pred_cls)
            print("2")
            test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
            print("train_acc")
            print(test_acc)
            # print("train_avg_acc")
            # print(avg_per_class_acc)
            test_true_seg = np.concatenate(test_true_seg, axis=0)
            test_pred_seg = np.concatenate(test_pred_seg, axis=0)
            test_iou = calculate_sem_IoU(test_pred_seg, test_true_seg)
            # outstr="Test :: test area :%s , test acc :%.6f , test iou : %.6f" %(test_area,
            #                                                                     accuracy,
            #                                                                     np.mean(test_iou))
            iou = np.mean(test_iou)

            outstr = "Test :: test area :%s " % (test_area)
            #print(outstr)
            print(iou)
            # print(outstr)
            all_true_cls.append(test_true_cls)
            all_pred_cls.append(test_pred_cls)
            all_true_seg.append(test_true_seg)
            all_pred_seg.append(test_pred_seg)

                # seg_pred = seg_pred.transpose(0, 2, 1)
                # argmax = ops.ArgMaxWithValue(axis=2)
                # index, pred = argmax(seg_pred)
                # pred_np = index
                # seg_np = target

if __name__=="__main__":
    # Testing setting
    parser = argparse.ArgumentParser(description="Testing for DGCNNDense S3DSIS")
    parser.add_argument("--path", type=str, default='/home/cxh/dataset/data/stanford_indoor3d')
    parser.add_argument("--pretrain_path", type=str, default="/home/cxh/文档/seu3dlab/ms3d/example/dgcnn/checkpoint/seg/")
    parser.add_argument("--test_area", type=str, default="1", choices=["1", "2", '3', '4', '5', '6', '7'])
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--num_classes", type=int, default=13)
    #parser.add_argument("--test_area", type=str, default="1", choices=["1", "2", '3', '4', '5', '6', '7'])
    args=parser.parse_args()
    eval_net(args)
