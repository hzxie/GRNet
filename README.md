# GRNet

This repository contains the source code for the paper [GRNet: Gridding Residual Network for Dense Point Cloud Completion](https://arxiv.org/abs/2006.03761).

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/hzxie/GRNet.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/hzxie/GRNet/context:python)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/hzxie/GRNet.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/hzxie/GRNet/alerts/)

![Overview](https://infinitescript.com/wordpress/wp-content/uploads/2020/07/GRNet-Overview.png)

## Cite this work

```
@inproceedings{xie2020grnet,
  title={GRNet: Gridding Residual Network for Dense Point Cloud Completion},
  author={Xie, Haozhe and 
          Yao, Hongxun and 
          Zhou, Shangchen and 
          Mao, Jiageng and 
          Zhang, Shengping and 
          Sun, Wenxiu},
  booktitle={ECCV},
  year={2020}
}
```

## Datasets

We use the [ShapeNet](https://www.shapenet.org/), [Compeletion3D](http://completion3d.stanford.edu/), and [KITTI](http://www.cvlibs.net/datasets/kitti/) datasets in our experiments, which are available below:

- [ShapeNet](https://drive.google.com/drive/folders/1P_W1tz5Q4ZLapUifuOE4rFAZp6L1XTJz)
- [Completion3D](http://download.cs.stanford.edu/downloads/completion3d/dataset2019.zip)
- [KITTI](https://drive.google.com/drive/folders/1fSu0_huWhticAlzLh3Ejpg8zxzqO1z-F)

## Pretrained Models

The pretrained models on ShapeNet are available as follows:

- [GRNet for ShapeNet](https://gateway.infinitescript.com/?fileName=GRNet-ShapeNet.pth) (306.8 MB)
- [GRNet for KITTI](https://gateway.infinitescript.com/?fileName=GRNet-KITTI.pth) (306.8 MB)

## Prerequisites

#### Clone the Code Repository

```
git clone https://github.com/hzxie/GRNet.git
```

#### Install Python Denpendencies

```
cd GRNet
pip install -r requirements.txt
```

#### Build PyTorch Extensions

**NOTE:** PyTorch >= 1.4, CUDA >= 9.0 and GCC >= 4.9 are required.

```
GRNET_HOME=`pwd`

# Chamfer Distance
cd $GRNET_HOME/extensions/chamfer_dist
python setup.py install --user

# Cubic Feature Sampling
cd $GRNET_HOME/extensions/cubic_feature_sampling
python setup.py install --user

# Gridding & Gridding Reverse
cd $GRNET_HOME/extensions/gridding
python setup.py install --user

# Gridding Loss
cd $GRNET_HOME/extensions/gridding_loss
python setup.py install --user
```

#### Preprocess the ShapeNet dataset

```
cd $GRNET_HOME/utils
python lmdb_serializer.py /path/to/shapenet/train.lmdb /path/to/output/shapenet/train
python lmdb_serializer.py /path/to/shapenet/valid.lmdb /path/to/output/shapenet/val
```

You can download the processed ShapeNet dataset [here](https://gateway.infinitescript.com/?fileName=ShapeNetCompletion).

#### Update Settings in `config.py`

You need to update the file path of the datasets:

```
__C.DATASETS.COMPLETION3D.PARTIAL_POINTS_PATH    = '/path/to/datasets/Completion3D/%s/partial/%s/%s.h5'
__C.DATASETS.COMPLETION3D.COMPLETE_POINTS_PATH   = '/path/to/datasets/Completion3D/%s/gt/%s/%s.h5'
__C.DATASETS.SHAPENET.PARTIAL_POINTS_PATH        = '/path/to/datasets/ShapeNet/ShapeNetCompletion/%s/partial/%s/%s/%02d.pcd'
__C.DATASETS.SHAPENET.COMPLETE_POINTS_PATH       = '/path/to/datasets/ShapeNet/ShapeNetCompletion/%s/complete/%s/%s.pcd'
__C.DATASETS.KITTI.PARTIAL_POINTS_PATH           = '/path/to/datasets/KITTI/cars/%s.pcd'
__C.DATASETS.KITTI.BOUNDING_BOX_FILE_PATH        = '/path/to/datasets/KITTI/bboxes/%s.txt'

# Dataset Options: Completion3D, ShapeNet, ShapeNetCars, KITTI
__C.DATASET.TRAIN_DATASET                        = 'ShapeNet'
__C.DATASET.TEST_DATASET                         = 'ShapeNet'
```

## Get Started

To train GRNet, you can simply use the following command:

```
python3 runner.py
```

To test GRNet, you can use the following command:

```
python3 runner.py --test --weights=/path/to/pretrained/model.pth
```

## License

This project is open sourced under MIT license.
