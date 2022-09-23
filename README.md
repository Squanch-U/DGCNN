# DGCNN-mindspore
This repo is the mindspore implementation of **Dynamic Graph CNN for Learning on Point Clouds (DGCNN)**(https://arxiv.org/pdf/1801.07829). Our code skeleton is borrowed from [WangYueFt/dgcnn](https://github.com/WangYueFt/dgcnn/tree/master/pytorch).

&nbsp;
## Requirements
- Ubuntu 20.04
- Python 3.7
- mindspore=1.8.1
- CUDA 11.1
- Package: h5py, sklearn, plyfile

&nbsp;
## Point Cloud Classification
### Run the training script:
``` 
python dgcnn_cls_train.py
```
### Run the evaluation script after training finished:
``` 
python dgcnn_cls_eval.py
```
### Performance:
ModelNet40 dataset

|  | Mean Class Acc | Overall Acc |
| :---: | :---: | :---: |
| Paper (1024 points) | 90.2 | 92.9 |
| This repo (1024 points) | **  ** | **  ** |
