from __future__ import print_function
import sys

import argparse
import os
import random
import math
import mindspore.numpy as np
import mindspore
from mindspore import load_checkpoint,load_param_into_net,context
import mindspore.dataset as ds
import mindspore.ops as ops
from mindvision.ms3d.dataset.S3DIS import S3DISD
from mindvision.ms3d.models.dgcnn_train_seg import Dgcnn
from tqdm import tqdm
from mindspore import nn,Tensor
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



def S3DIS_test(args):
    print("========starting=============")
    all_true_cls=[]
    all_pred_cls=[]
    all_true_seg=[]
    all_pred_seg=[]
    for test_area in range(1,7):
        visual_file_index=0
        test_area=str(test_area)
        if os.path.exists("/mindvision/ms3d/dataset/data/indoor3d_sem_seg_hdf5_data_test/room_filelist.txt"):
            with open("/mindvision/ms3d/dataset/data/indoor3d_sem_seg_hdf5_data_test/room_filelist.txt") as f:
                for line in f:
                    if (line[5]) == test_area:
                        break
        if (args.test_area=="all") or (test_area==args.test_area):
            test_dataset_generator = S3DISD(path=args.path,
                                            split="val",
                                            batch_size=1,
                                            repeat_num=1,
                                            shuffle=True,
                                            download=False,
                                            test_area=test_area)
            test_dataset_generator = test_dataset_generator.run()
            itr = test_dataset_generator.create_tuple_iterator()
            test_dataloader = ds.GeneratorDataset(itr, ["data", "label"], shuffle=True)

            model= Dgcnn()
            path=os.path.join(args.pretrain_path, 'model_%s.ckpt' % test_area)
            print(path)
            param_dict = load_checkpoint(path)
            load_param_into_net(model, param_dict)
            print("sucessfully load pretrain model")
            model.set_train(False)

            test_acc=0.0
            count=0.0
            test_true_cls=[]
            test_pred_cls=[]
            test_true_seg=[]
            test_pred_seg=[]
            for _, data in tqdm(enumerate(test_dataloader.create_dict_iterator(), 0)):
                points, target = data['data'], data['label']
                seg_pred=model(points)
                seg_pred = seg_pred.transpose(0, 2, 1)
                argmax = ops.ArgMaxWithValue(axis=2)
                index, pred = argmax(seg_pred)
                pred_np = index
                seg_np=target

                test_true_seg.append(seg_np)
                test_pred_seg.append(pred_np)

                test_true_cls.append(seg_np.reshape(-1))
                test_pred_cls.append(pred_np.reshape(-1))
            test_true_cls=np.concatenate(test_true_cls)
            test_pred_cls=np.concatenate(test_pred_cls)

            test_acc = metrics.accuracy_score(test_true_cls, test_pred_cls)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true_cls, test_pred_cls)
            print("train_acc")
            print(test_acc)
            #print("train_avg_acc")
            #print(avg_per_class_acc)
            test_true_seg=np.concatenate(test_true_seg,axis=0)
            test_pred_seg=np.concatenate(test_pred_seg,axis=0)
            test_iou=calculate_sem_IoU(test_pred_seg,test_true_seg)
            # outstr="Test :: test area :%s , test acc :%.6f , test iou : %.6f" %(test_area,
            #                                                                     accuracy,
            #                                                                     np.mean(test_iou))
            iou=np.mean(test_iou)

            outstr = "Test :: test area :%s " % (test_area)
            print(outstr)
            print(iou)
            #print(outstr)
            all_true_cls.append(test_true_cls)
            all_pred_cls.append(test_pred_cls)
            all_true_seg.append(test_true_seg)
            all_pred_seg.append(test_pred_seg)
    if args.test_area == 'all':
        all_true_cls=np.concatenate(all_true_cls)
        all_pred_cls=np.concatenate(all_pred_cls)

        # all_test_acc = nn.Accuracy("classification")
        # all_test_acc.clear()
        # all_test_acc.update(all_true_cls, all_pred_cls)
        # accuracy = all_test_acc.eval()

        all_true_seg=np.concatenate(all_true_seg,axis=0)
        all_pred_seg=np.concatenate(all_pred_seg,axis=0)
        all_ious=calculate_sem_IoU(all_pred_seg,all_true_seg)
        #outstr="Ovarall Test :: test acc : %.6f,test iou : %.6f" %(accuracy,np.mean(all_ious))
        #outstr = "Ovarall Test :: test iou : %.6f" % (np.mean(all_`1   ious))
        print("Overall test")
        print(np.mean(all_ious))

if __name__=="__main__":
    #Testing setting
    parser=argparse.ArgumentParser(description="Testing for DGCNNDense S3DSIS")
    parser.add_argument("--path",type=str,default='/home/cxh/dataset/data/stanford_indoor3d')
    parser.add_argument("--pretrain_path",type=str,default="/home/cxh/ZZ/")
    parser.add_argument("--test_area",type=str,default=" 0",choices=["1","2",'3','4','5','6','7'])
    # args=parser.parse_args()
    # S3DIS_test(args)

