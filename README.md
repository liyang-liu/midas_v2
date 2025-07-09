# Contents

<!-- TOC -->

- [Midas Description](#midas-description)
- [Model Structure](#model-structure)
- [Dataset](#Dataset)
- [Features](#Features)
    - [Mixed Precision](#mixed-precision)
- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Scripts](#Scripts)
    - [Scripts and Example Code](#scripts-and-example-code)
    - [Scripts Parameters](#scripts-parameters)
    - [Training Process](#training-process)
    - [Validation Prcoess](#validation-process)
- [Inference Process](#inference-process)
- [Model Description](#model-description)
    - [Features](#Features)
        - [Feature Evaluation](#feature-evaluation)
- [Explanation on Stochastic Situation](#explanation-on-stochastic-situation)
- [ModelZoo HomePage](#modelZoo-homePage)

<!-- /TOC -->

# Midas Description

## Summary

Midas is the codename for "Towards Robust Monocular Depth Estimation:Mixing Datasets for Zero-shot Cross-dataset Transfer",
provides estimation of image depth，it uses 5 different datasets for training，
these mixed-type strategy helps to achieve multi-objective optimization.
One of the datasets is a self-made 3D movie dataset. And it uses 6 datasets totally different from training for validaiton.
This repo only uses the RedWeb dataset for training.
For detailed description of the model network, please refer to [Towards Robust Monocular Depth Estimation:Mixing Datasets for
Zero-shot Cross-dataset Transfer](https://arxiv.org/pdf/1907.01341v3.pdf)，Midas模型网络的Pytorch版本实现，可参考(<https://github.com/intel-isl/MiDaS>)

Midas全称为 "" 用来估计图片的深度信息, 使用了五个不同的训练数据集，五个训练数据集混合策略为多目标优. 其中包括作者自制的3D电影数据集. 使用6个和训练集完全不同的测试集进行验证。
本次只使用ReDWeb数据集进行训练。Midas模型网络具体细节可参考 。

## Paper

1. [Paper:](https://arxiv.org/pdf/1907.01341v3.pdf) Ranftl*, Katrin Lasinger*, David Hafner, Konrad Schindler, and Vladlen Koltun.

# Model Structure

Overall Model structure as described in：
[Link](https://arxiv.org/pdf/1907.01341v3.pdf)

# Dataset

Dataset used：[ReDWeb](<https://www.paperswithcode.com/dataset/redweb>)

- Dataset size：
    - training size：292M, 3600 images
- Data format：
    - original imgs：JPG
    - Depth RDs：PNG

# Features

## Mixed Precision

Adopts [MixedPrecision_English](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html) training method,
by using Full Purity (FP32) and Half Purity (FP16) to significantly improve Deep Learning efficiency and memory usage,
yet achieving sufficient accuracy. 
This results in training of even larger models or larger batch size in dedicated hardware. 
The user is recommended to checkout out the "INFO" diary, and search for "reduce precision" for calculations 
that use reduced purity.

采用[混合精度](https://www.mindspore.cn/tutorials/experts/zh-CN/master/others/mixed_precision.html)的训练方法
5使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。
混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型
或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。
用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

 
# System Requirements 

- Hardware(GPU)
    - Prepare GPU environment
      - Install CUDA 11.6.0, cudnn 8.4.1.50, TensorRT 8.4.2.4(optional), then set following
      ```text
      $ export PATH=/usr/local/cuda-11.6/bin:$PATH
      $ export LD_LIBRARY_PATH=/usr/local/cuda-11.6/lib64:$LD_LIBRARY_PATH
      $ export CUDA_HOME=/usr/local/cuda-11.6
      ```
      
      - create conda environment to install python 3.9.11
      ```text
      $ conda create -n midas_v2 python=3.9.11 -y
      $ conda activate midas_v2
      $ pip install h5py
      $ pip install pip install opencv-python==4.9.0.80
      ```      
      
      
- Installation
    - Install Mindspore: if cuda was configured correctly, should automatically download mindspore with GPU support)
      ```text
      $ pip install mindspore-dev -i https://pypi.tuna.tsinghua.edu.cn/simple
      ```
      
    - Verify by running following command
      ```text    
      $ python -c "import mindspore;mindspore.set_device(device_target='GPU');mindspore.run_check()"
      ```
      ```shell
	MindSpore version: __version__
	The result of multiplication calculation is correct, MindSpore has been installed on platform [GPU] successfully!      
      ```
      
    - For details, see [MindSpore](https://www.mindspore.cn/install/en)
    
- If detailed information is needed, please check following resources 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# Quick Start
After installing MindSpore from its official website, follow steps below for training and validation:

- Pretrain models 

  Before training starts, need to obtain mindspore ImageNet pre-train models, use pre-trained model from resnext101, 
  model name [resnext101_32x8d_wsl](<https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth>),
  after downloading the "pth" file, run `python src/utils/pth2ckpt.py /pth_path/ig_resnext101_32x8-c38310e5.pth`
  to convert the pth file to "ckpt" file.
  当开始训练之前需要获取mindspore图像网络预训练模型，使用在resnext101上训练出来的预训练模型[resnext101_32x8d_wsl](<https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth>),下载完pth文件之后,运行`python src/utils/pth2ckpt.py /pth_path/ig_resnext101_32x8-c38310e5.pth`将pth文件转换为ckpt文件.

- Preparation of dataset 


 The midas NN model uses ReDWeb Datasets for training, and uses Sintel, KITTI, TUM datasets for inference,
  dataset can be downloaded from [ReDWeb](<https://www.paperswithcode.com/dataset/redweb>),[Sintel](http://sintel.is.tue.mpg.de),[Kitti](http://www.cvlibs.net/datasets/kitti/raw_data.php),[TUM](https://vision.in.tum.de/data/datasets/rgbd-dataset/download#freiburg2_desk_with_person).
  Specifically, Sintel dataset requires downloading of both original images and depth images, they should be saved into the "Sintel" dataset folder。
  For the TUM dataset: run the pre-processing function to generate associate.txt for data pairing。
  All pre-processing functions can be found in the "preprocess" folder，
  for details please checkout the "readme.md" document under the "preprocess" folder。

 midas网络模型使用ReDWeb数据集用于训练,使用Sintel,KITTI,TUM数据集进行推理,数据集可通过[ReDWeb](<https://www.paperswithcode.com/dataset/redweb>),[Sintel](http://sintel.is.tue.mpg.de),[Kitti](http://www.cvlibs.net/datasets/kitti/raw_data.php),[TUM](https://vision.in.tum.de/data/datasets/rgbd-dataset/download#freiburg2_desk_with_person)官方网站下载使用.
 Sintel数据集需要分别下载原图和深度图，放入到Sintel数据集文件夹中。TUM数据集根据处理函数得到associate.txt进行匹配数据。所有处理函数在preprocess文件夹下，具体可参考preprocess文件夹下的readme.md。
  
- After downloading datasets, save the dataset and code under the folder structure shown below, 
  and save `midas/mixdata.json` and datasets under `data/`.
  下载完数据集之后,按如下目录格式存放数据集和代码, 并将`midas/mixdata.json`和数据集放到`data/`目录下即可:

    ```path
        └── midas
        └── data
            ├── mixdata.json
            ├── ReDWeb_V1
            |   ├─ Imgs
            |   └─ RDs
            ├── Kitti_raw_data
            |   ├─ 2011_09_26_drive_0002_sync
            |   |   ├─depth
            |   |   └─image
            |   ├─ ...
            |   |   ├─depth
            |   |   └─image
            |   ├─ 2011_10_03_drive_0047_sync
            |   |   ├─depth
            |   |   └─image
            ├── TUM
            |   ├─ rgbd_dataset_freiburg2_desk_with_person
            |   |   ├─ associate.txt
            |   |   ├─ depth.txt
            |   |   ├─ rgb.txt
            |   |   ├─ rgb
            |   |   |   ├─ rgb
            |   |   |   ├─ 1311870426.504412.png
            |   |   |   ├─ ...
            |   |   |   └─ 1311870426.557430.png
            |   |   |   ├─ depth
            |   |   |   ├─ 1311870427.207687.png
            |   |   |   ├─ ...
            |   |   |   └─ 1311870427.376229.png
            ├── Sintel
            |   ├─ depth
            |   ├─ final_left
            |   └─ occlusions
    ```

- Usage under the "Ascend" Processing Unit Environment Ascend处理器环境运行

```text
# Distributed training
Command: bash run_distribute_train.sh 8 ./ckpt/midas_resnext_101_WSL.ckpt


# Single Machine Unit training
Command：bash run_standalone_train.sh [DEVICE_ID] [CKPT_PATH]

# Running Evluation Example
Command：bash run_eval.sh [DEVICE_ID] [DATA_NAME] [CKPT_PATH]
```

# Scripts

## Scripts and Example Code
310 refers to a Ascend computing unit processor developed from Huawei [310](https://www.mindspore.cn/docs/programming_guide/en/r1.3/multi_platform_inference_ascend_310.html), we will NOT use it in this repo.

```shell

└──midas
  ├── README.md
  ├── ascend310_infer
    ├── inc
        └── utils.sh                       # 310 head script
    ├── src
        ├── main.cc                        # 310 main function
        └── utils.cc                       # 310 function
    ├── build.sh                           # build 310 envrionment
    └── CMakeLists.txt                     # 310 make file
  ├── scripts
    ├── run_distribute_train.sh            # start Ascend distributed training（8 cards）
    ├── run_eval.sh                        # start Ascend validation
    ├── run_standalone_train.sh            # start Ascend single machine（stand alone）
    ├── run_train_gpu.sh                   # start GPU training
    └── run_infer_310.sh                   # start Ascend 310 inference
  ├── src
    ├── utils
        ├── loadImgDepth.py                # load dataset
        └── transforms.py                  # convert images
    ├─config.py                            # training configuration
    ├── cunstom_op.py                      # NN operation
    ├── blocks_ms.py                       # NN components
    ├── loss.py                            # loss function
    ├── util.py                            # image io tool 
    └── midas_net.py                       # main net definition
  ├── config.yaml                          # training parameters configuration
  ├── midas_eval.py                        # NN evaluation
  ├── midas_export.py                      # model export
  ├── midas_run.py                         # model run
  ├── postprocess.py                       # 310 pose-pcocessing
  └── midas_train.py                       # train nn model
```

## Training Parameters
Configure settings in file "config.yaml"


- Configure training parameters:

```python
device_target: 'Ascend'                                          #processor type,accept type: CPU,GPU,Ascend
device_id: 7                                                     #device ID
run_distribute: False                                            #whether to use distributed training
is_modelarts: False                                              #whether to rain in cloud
no_backbone_params_lr: 0.00001                                   #1e-5
no_backbone_params_end_lr: 0.00000001                            #1e-8
backbone_params_lr: 0.0001                                       #1e-4
backbone_params_end_lr: 0.0000001                                #1e-7
power: 0.5                                                       #PolynomialDecayLR seed control: lr parameter 种控制lr参数
epoch_size: 400                                                  #total epoch
batch_size: 8                                                    #batch_size
lr_decay: False                                                  #whether use dynamically adjusted learn rate 
train_data_dir: '/midas/'                                        #dataset root path
width_per_group: 8                                               #network parameter
groups: 32
in_channels: 64
features: 256
layers: [3, 4, 23, 3]
img_width: 384                                                   #NN input image width
img_height: 384                                                  #NN input image height
nm_img_mean: [0.485, 0.456, 0.406]                               #Image pre-processing regularization parameter
nm_img_std: [0.229, 0.224, 0.225]
keep_aspect_ratio: False                                         #keep aspect ratio
resize_target: True                                              #if True, modify size of image, mask, target，otherwise only modify image size
ensure_multiple_of: 32                                           #ensure image size is multiple of 32 
resize_method: "upper_bound"                                     #resize模式
```

- Configure validation parameters: 

```python
datapath_TUM: '/data/TUM'                                        #TUM dataset path
datapath_Sintel: '/data/sintel/sintel-data'                      #Sintel dataset path
datapath_ETH3D: '/data/ETH3D/ETH3D-data'                         #ETH3D dataset path
datapath_Kitti: '/data/Kitti_raw_data'                           #Kitti dataset path
datapath_DIW: '/data/DIW'                                        #DIW dataset path
datapath_NYU: ['/data/NYU/nyu.mat','/data/NYU/splits.mat']       #NYU dataset path
ann_file: 'val.json'                                             #path to save inference results
ckpt_path: '/midas/ckpt/Midas_0-600_56_1.ckpt'                   #path to save inference ckpt file
data_name: 'all'                                                 #dataset for inference, including Sintel,Kitti,TUM,DIW,ETH3D,all
```

- Configure running and model export parameter

```python
input_path: '/midas/input'                      #input image path
output_path: '/midas/output'                    #model output path 模型输出图片的路径
model_weights: '/ckpt/Midas_0-600_56_1.ckpt'    #model parameter path
file_format: "MINDIR"  # ["AIR", "MINDIR"]      #AIR/MIDIR
```

## Training Process

### Usage

#### For Ascend Envionrment (Ascend处理器环境运行)

```text
# Distributed Training
Command：bash run_distribute_train.sh 8 ./ckpt/midas_resnext_101_WSL.ckpt
# Single Machine Training
Command：bash run_standalone_train.sh [DEVICE_ID] [CKPT_PATH]
# Run Evaluation Example
Command：bash run_eval.sh [DEVICE_ID] [DATA_NAME] [CKPT_PATH]

```

#### For GPU Environment   (GPU处理器环境运行)

```text
Command：bash run_train_GPU.sh [DEVICE_NUM] [DEVICE_ID] [CKPT_PATH]
# Distributed training
Command：bash run_train_GPU.sh 8 0,1,2,3,4,5,6,7 /ckpt/midas_resnext_101_WSL.ckpt
# Single machine training
Command：bash run_train_GPU.sh 1 0 /ckpt/midas_resnext_101_WSL.ckpt
# Running evaluation example
Command：bash run_eval.sh [DEVICE_ID] [DATA_NAME] [CKPT_PATH]

```
Or

```text
cd midas
mv mida_pth.ckpt ckpt
bash -x scripts/run_train_gpu.sh 1 0 ckpt/midas_pth.ckpt
# This creates a folder train_GPU in midas, and produces some error messages in train_GPUS/train.log
# Try re-run the command to see detailed error message
python3 /home/liyang/Work/Research/Depth-for-Anything/Midas/midas/scripts/../midas_train.py --device_target GPU --run_distribute False --device_id 0 --model_weights ckpt/midas_pth.ckpt
```
### Results

- Train midas with ReDWeb datasets

```text
Distributed training results:（8P）
epoch: 1 step: 56, loss is 579.5216
epoch time: 1497998.993 ms, per step time: 26749.982 ms
epoch: 2 step: 56, loss is 773.3644
epoch time: 74565.443 ms, per step time: 1331.526 ms
epoch: 3 step: 56, loss is 270.76688
epoch time: 63373.872 ms, per step time: 1131.676 ms
epoch: 4 step: 56, loss is 319.71643
epoch time: 61290.421 ms, per step time: 1094.472 ms
...
epoch time: 58586.128 ms, per step time: 1046.181 ms
epoch: 396 step: 56, loss is 8.707727
epoch time: 63755.860 ms, per step time: 1138.498 ms
epoch: 397 step: 56, loss is 8.139318
epoch time: 47222.517 ms, per step time: 843.259 ms
epoch: 398 step: 56, loss is 10.746628
epoch time: 23364.224 ms, per step time: 417.218 ms
epoch: 399 step: 56, loss is 7.4859796
epoch time: 24304.195 ms, per step time: 434.003 ms
epoch: 400 step: 56, loss is 8.2024975
epoch time: 23696.833 ms, per step time: 423.158 ms
```

## Validation Process

### Usage

#### Ascend processing environment (Ascend处理器环境运行)
Specify inference dataset in file "config.yaml" for field "data_name". Default: all datasets 

```bash
# Validation
bash run_eval.sh [DEVICE_ID] [DATA_NAME]
```

### Results

Open file "val.json" to check inference results, example below：

```text
{"Kitti": 24.222 "Sintel":0.323 "TUM":15.08 }
```

#### GPU Environment 

```bash
# Validaton
bash run_eval.sh [DEVICE_ID] [DATA_NAME] [CKPT_PATH] [DEVICE_TARGET]
```

### Results

Open file "val.json" to check inference results, as shown below ：

```text
{"Kitti": 24.222 "Sintel":0.323 "TUM":15.08 }
```

# Inference Process

## Export MindIR

```shell
python midas_export.py
```

Configure settings in file "config.yam" 

### Perform Inference on Ascend 310 (在Ascend310执行推理)
Before running inference, you must generate the "mindir" file by running `midas_export.py`. 
The demo below illustrates how to perform inference using the `mindir` model. 

```shell
# Ascend310 inference
bash run_infer_310.sh [MODEL_PATH] [DATA_PATH] [DATASET_NAME] [DEVICE_ID]
```

- `MODEL_PATH` mindir file path
- `DATA_PATH`  inference dataset path 
- `DATASET_NAME` inference dataset name，should be Kitti，TUM，Sintel。
- `DEVICE_ID`   optional，default value is 0。

### Results
Inference results are stored in the working path of the script, you can checkout precision calculation results under file `result_val.json`.

```text
{"Kitti": 18.27 "Sintel":0.314 "TUM":13.27 }
```

# Model Description

## Features

### Feature Evaluation

#### Feature from ReDWeb (ReDWeb性能参数)

| Parameters          | Ascend 910                   |V100-PCIE                  |
| ------------------- | ---------------------------- |--------------------------- |
| Model version       | Midas                        | Midas                      |
| resources           | Ascend 910；CPU：2.60GHz，192 core；memeory：755G                  | Tesla V100-PCIE 32G ， cpu  52cores 2.60GHz，RAM 754G                 |
| upload time         | 2021-06-24                   |2021-11-30                  |
| MindSpore version   | 1.2.0                        |1.6.0.20211125              |
| Dataset             | ReDWeb                       |ReDWeb                      |
| Pre-trained model   | ResNeXt_101_WSL              |ResNeXt_101_WSL             |
| Training parameters | epoch=400, batch_size=8, no_backbone_lr=1e-4,backbone_lr=1e-5   | epoch=400, batch_size=8, no_backbone_lr=1e-4,backbone_lr=1e-5   |
| Optimizer           | Adam                         |Adam                        |
| Loss function       | Self-defined loss function   | 自定义损失函数               |
| Speed               | 8pc: 423.4 ms/step           |8pc: 920 ms/step  1pc:655ms/step      |
| Train metric        | "Kitti": 24.222 "Sintel":0.323  "TUM":15.08    |"Kitti"：23.870 "sintel": 0.322569 "TUM": 16.198  |

# ModelZoo HomePage

 Please checkout their official homepage[homepage](https://gitee.com/mindspore/models)。
