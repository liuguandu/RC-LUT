## Reconstructed Convolution Module Based Look-Up Tables for Efficient Image Super-Resolution

[Guandu Liu*], Yukang Ding, Mading Li, Ming Sun, Xing Wen and [Bin Wang#]

## Efficiency
![image](./doc/psnr_volume.png#pic_center=50x50)
## Overview
The core idea of our paper is RC Module.
![image](./doc/overview.png)

## Usage
Our code follows the architecture of [MuLUT](https://github.com/ddlee-cn/MuLUT). In the sr directory, we provide the code of training RC-LUT networks, transferring RC-LUT network into LUts, finetuning LUTs, and testing LUTs, taking the task of single image super-resolution as an example.
In the `common/network.py`, `RC_Module` is the core module of our paper.
### Dataset

Please following the instructions of [training](./data/DIV2K/README.md). And you can also prepare [SRBenchmark](./data/DIV2K/README.md)
### Installation
Clone this repo
```
git clone https://github.com/liuguandu/RC-LUT
```
Install requirements: torch>=1.5.0, opencv-python, scipy
### Train
First, please train RC network follow next code
```
sh ./sr/5x57x79x9MLP_combined.sh
```

updating...