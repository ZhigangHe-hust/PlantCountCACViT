<p align="center">
    <h1 align="center">PlantCountCACViT</h1>
</p>

![image text](https://github.com/ZhigangHe-hust/PlantCountCACViT/blob/main/Figs/fig1.png)

<div align="center">
  <a href="./README.md">English</a> | 
  <a href="./README_CN.md">简体中文</a>
</div>


## 1.Introduction

Due to the large variety of plant varieties and the continuous selection of new varieties every year, and the conventional counting algorithm needing to predict the category of the counting object, collect data, label data, develop models, train models, and evaluate the counting results for this category, there is an urgent need for a variety-independent plant counting method. In recent years, category-independent counting does not require prior knowledge or identification of the specific category of an object, it does not rely on the category information of the object to be counted, but focuses on learning how to count. Based on the CACViT model pre-trained on FSC147, this project was trained and fine-tuned on the collected plant datasets to obtain PlantCountCACViT, an efficient and accurate species-independent plant counting model that can be applied to the real open environment across scales, scenarios, and time and space.

## 2.Datasets and pretrained model

FSC147 dataset download: Reference<a href="https://github.com/cvlab-stonybrook/LearningToCountEverything/tree/master" title="Learning To Count Everything">Learning To Count Everything</a>

The plant count dataset of this project is downloaded: <a href="https://pan.quark.cn/s/76cec041ff98" title="PlantCountDataset">PlantCountDataset,</a> Extraction code: **8Gnp**

This project is pre-trained on the basis of the optimal model of CACViT, and the download of the pre-trained model file can be found in <a href="https://github.com/Xu3XiWang/CACViT-AAAI24" title="CACViT">CACViT</a>.

## 3.Installation

**Python：**3.8.18

**GPU：**RTX3090

```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/
whl/torch_stable.html
```

```
pip install -r requirements.txt
```

## 4.Training and Evaluation

### Training

```bash
python train_val.py
```

**Resume：**

Determine the file path and start_epoch of the model you wish to resume training based on the actual situation.

```
python train_val.py --resume ./output_dir/checkpoint-30.pth --start_epoch 30
```

### Evaluation

```bash
python test.py
```

## 4.Conclusion

<a href="https://pan.quark.cn/s/aaa63b751b19" title="Best Model">Best Model</a>，Extraction code：**QDVx**

The test results of the Best Model on the testdata of the PlantCountDataset:
<div align="center">

<table>
  <thead>
    <tr>
      <th align="center">Metrics</th>
      <th align="center">Values</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">MAE</td>
      <td align="center">5.25</td>
    </tr>
    <tr>
      <td align="center">RMSE</td>
      <td align="center">7.37</td>
    </tr>
    <tr>
      <td align="center">rMAE</td>
      <td align="center">0.42</td>
    </tr>
    <tr>
      <td align="center">rRMSE</td>
      <td align="center">0.44</td>
    </tr>
    <tr>
      <td align="center">R2</td>
      <td align="center">0.95</td>
    </tr>
    <tr>
      <td align="center">Average FLOPs per sample</td>
      <td align="center">449.8607 GFLOPs</td>
    </tr>
    <tr>
      <td align="center">Testing time</td>
      <td align="center">57s</td>
    </tr>
  </tbody>
</table>

</div>
