# ETOG

## Introduction
This is the official implementation code base for [A Parameter-Efficient Tuning Framework for Language-guided Object Grounding and Robot Grasping](https://arxiv.org/pdf/2409.19457) accepted for ICRA 2025. Our project produces **three separated** github repos for ETOG, ETRG-A, ETRG-B models. Stay tuned for code release.

This git repo includes the ETOG model, which is designed for parameter-efficient tuning on the Referring Expression Segmentation (RES) task.

![Pipeline Image](pipeline.png)

## Implementations for RGS and RGA Robotics Tasks

1. The ETRG-A model designed for ```Referring Grasp Synthesis (RGS)``` task can be found here.

2. The ETRG-B model designed for ```Referring Grasp Affordance (RGA)``` task can be found here.


## Preparation
1. Conda env: We used Pytorch (2.1.0+cu118), other packages are in ```requirements.txt```
2. Refcoco related dataset
   - The detailed instruction is in [prepare_datasets.md](tools/prepare_datasets.md)
   - The folder arrangement after preparation should be like this:

```
$ETOG
├── config
├── model
├── enging
├── pretrain (manually download from CLIP -> R50, R101, ViT-B-16)
├── tools
│     ├── data_process.py
│     └── ...
├── ...
└── datasets
    ├── anns
    ├── lmdb
    │   ├── refcoco  
    │   ├── refcoco+
    │   ├── refcocog
    │   └── ...
    ├── masks
    │   ├── refcoco  
    │   ├── refcoco+
    │   ├── refcocog
    │   └── ...
    └── images

```

## Pretrianed model wegihts and training/testing logs
Performance (mIoU) on Refcoco dataset:

| Backbone | val | test A | test B | Weights| Train log | Test log |
| ---- |:-------------:| :-----:|:-----:|:-----:|:-----:|:-----:|
| CLIP-R50 | 72.31  | 75.49 | 66.62 | [models](https://drive.google.com/file/d/1PKhFIGmwyl5O2maI8OoLzHoH_g8-iWPK/view?usp=drive_link) | [log](https://drive.google.com/file/d/1tQAKs1U99we41b5aYy2s0U4VJzPlUy6O/view?usp=drive_link) | [log](https://drive.google.com/file/d/1yqdjzhrthWdJh1hLepqw7f2llPTSLzZo/view?usp=drive_link) |
| CLIP-R101 | 73.37 | 76.16 | 68.54 | [models](https://drive.google.com/file/d/18gUcryjxEmBrCXGjGvu7m4nWb3aJisx0/view?usp=drive_link) | [log](https://drive.google.com/file/d/1IBf-V-InMyO6idr1knptc5dd2v87m14M/view?usp=drive_link) | [log](https://drive.google.com/file/d/14p0E69veYk0qoylbyXa1hkTtuPD8iRyG/view?usp=drive_link) |
| CLIP-ViT-B| 73.37 | 76.90 | 69.34 | [models](https://drive.google.com/file/d/1xOTsdjR4HuknS1VdRSCFqnLtZ2HzK21N/view?usp=drive_link) | [log](https://drive.google.com/file/d/1ApbLv2IKq1Q_IvVKvwMS6ksuKp5xucfW/view?usp=drive_link) | [log](https://drive.google.com/file/d/13bDOxfSoePXqmsyyJVH37BWKNQCEF7gA/view?usp=drive_link) |

We release all Refcoco-related pretrained weights reported on our paper. 

More training/testing logs and model weights available for Refcoco+ and Refcocog benchmarks are available [here](https://drive.google.com/drive/folders/1NDkopub0oL_WTm3TqS4s3htsqPYUTRk9?usp=sharing) on our google drive.

## Train ETOG:

Quick run

```
bash run_scripts/train.sh
```
Please modify the config files ```(e.g. config/refcoco/bridge_r50.ymal)``` to change the batch_size, directory and test-split etc. values.

Our defualt setup: bs=16 on 1 NVIDIA RTX 2080 TI GPU.

## Test ETOG:

Quick run

```
bash run_scripts/test.sh
```
or directly run ```test.py``` while changing the ```--config directory```

We also provide prediction visualization saving functionality by setting up

```
TEST: 
  visualizate: True
```

in ```.yaml``` files. Currently, we support attention viusalizations for R50 and R101 (but not ViT backbone) in heatmap style.

![Pipeline Image](attention_map.png)

## Acknowledgment

The code is heavily adapted from [ETIRS](https://github.com/kkakkkka/ETRIS/tree/main). We appreciate the authors for their wonderful codebase.

## Citation

If ETOG is useful for your research, please consider citing:

```
@article{yu2024parameter,
  title={A Parameter-Efficient Tuning Framework for Language-guided Object Grounding and Robot Grasping},
  author={Yu, Houjian and Li, Mingen and Rezazadeh, Alireza and Yang, Yang and Choi, Changhyun},
  journal={arXiv preprint arXiv:2409.19457},
  year={2024}
}
```

