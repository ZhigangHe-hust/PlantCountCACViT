<p align="center">
    <h1 align="center">PlantCountCACViT</h1>
</p>

<div align="center">
  <a href="./README.md">English</a> | 
  <a href="./README_CN.md">简体中文</a>
</div>

## 一、简介

由于植物品种繁多，且每年还会不断选育新品种，而常规计数算法需要预知计数对象的类别，针对该类别收集数据、标注数据、开发模型、训练模型、评估计数结果，故技术上亟需一种品种无关的植物计数方法。近年来，类别无关计数无需事先知道或识别物体的具体类别，它不依赖于待计数对象的类别信息，而专注于学习如何计数。本项目基于在FSC147上预训练的CACViT模型，在收集的植物数据集上进行训练和微调，得到可适用真实开放环境下跨尺度、跨场景、跨时空的高效精准的品种无关植物计数模型PlantCountCACViT。

## 二、数据集及预训练模型文件下载

FSC147数据集下载：参考<a href="https://github.com/cvlab-stonybrook/LearningToCountEverything/tree/master" title="Learning To Count Everything">Learning To Count Everything</a>

本项目的植物计数数据集下载：<a href="https://pan.quark.cn/s/76cec041ff98"
title="PlantCountDataset">PlantCountDataset</a>，提取码：**8Gnp**

本项目在CACViT的最优模型的基础上进行预训练，预训练模型文件的下载可以参考<a href="https://github.com/Xu3XiWang/CACViT-AAAI24" title="CACViT">CACViT</a>

## 三、环境配置

**建议Python版本和显卡型号：**

Python 3.8.18，NVIDIA GeForce RTX 3090

**安装依赖：**

```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/
whl/torch_stable.html
```

```
pip install -r requirements.txt
```

## 四、任务实现过程

### 1.模型训练

```bash
python train_val.py
```

**断点续训：**

根据实际情况确定你恢复训练的模型权重文件路径和start_epoch

```
python train_val.py --resume ./output_dir/checkpoint-30.pth --start_epoch 30
```

### 2.模型测试

```bash
python test.py
```

## 四、结论

<a href="https://pan.quark.cn/s/aaa63b751b19" title="最优模型权重">最优模型权重</a>，提取码：**QDVx**

最优模型权重在本项目的植物计数数据集上的测试集上的测试结果为：

|         指标名称         |    指标数值     |
| :----------------------: | :-------------: |
|           MAE            |      5.25       |
|           RMSE           |      7.37       |
|           rMAE           |      0.42       |
|          rRMSE           |      0.44       |
|            R2            |      0.95       |
| Average FLOPs per sample | 449.8607 GFLOPs |
|       Testing time       |       57s       |

