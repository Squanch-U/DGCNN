# DGCNN-MindSpore
This repo is the mindspore implementation of **Dynamic Graph CNN for Learning on Point Clouds (DGCNN)**(https://arxiv.org/pdf/1801.07829). Our code skeleton is borrowed from [WangYueFt/dgcnn](https://github.com/WangYueFt/dgcnn/tree/master/pytorch).

&nbsp;
## Requirements
- Ubuntu 20.04
- Python 3.7
- MindSpore 1.8.1
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
| This repo (1024 points) |  |  |

&nbsp;
## Point Cloud Part Segmentation
### Run the training script:
``` 
python dgcnn_seg_train.py
```
### Run the evaluation script after training finished:
``` 
python dgcnn_seg_eval.py
```
### Performance:

ShapeNet part dataset

| | Mean IoU | Airplane | Bag | Cap | Car | Chair | Earphone | Guitar | Knife | Lamp | Laptop | Motor | Mug | Pistol | Rocket | Skateboard | Table
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | 
| Shapes | | 2690 | 76 | 55 | 898 | 3758 | 69 | 787 | 392 | 1547 | 451 | 202 | 184 | 283 | 66 | 152 | 5271 | 
| Paper | **85.2** | 84.0 | **83.4** | **86.7** | 77.8 | 90.6 | 74.7 | 91.2 | **87.5** | 82.8 | **95.7** | 66.3 | **94.9** | 81.1 | **63.5** | 74.5 | 82.6 |
| This repo |  |  |   |  |  |  | |  |  |  |  | |  |  |  |  |  |

### Ackonwledgements:
We thank a lot for the official repo of [WangYueFt/dgcnn](https://github.com/WangYueFt/dgcnn/tree/master/pytorch).

### License:
The code is released under MIT License(See LICENSE file for details)
