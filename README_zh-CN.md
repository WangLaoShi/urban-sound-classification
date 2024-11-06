# 城市声音分类

本项目的目标是构建一个神经网络，能够对 [Urban Sound 8k 数据集](https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz?download=1) 进行分类。<br>
项目的详细概述请参阅 [项目报告](https://github.com/WangLaoShi/urban-sound-classification/blob/main/report/report.pdf)。

## 项目结构

项目文件夹的结构如下：
- **data/** 包含处理后的数据和原始数据。要使用数据集重现结果，请将折叠文件夹放入 **data/raw/audio** 中，然后将元数据文件放入 **data/raw/metadata** 中。
- **models/** 包含训练好的模型，具体为项目中使用的缩放器和 PCA 模型。
- **notebooks/** 包含用于执行代码的 Jupyter notebooks。
- **src/** 包含 **data**、**model** 和 **utils** 子文件夹，其中包含与项目不同部分相关的代码。
- **report/** 包含用 LaTeX 编写的项目报告。

## 需求

项目中使用的库包括：*pandas*、*numpy*、*matplotlib*、*tensorflow*、*librosa*、*dask*、*keras_nightly*、*keras*、*scikit_learn*

您可以使用以下命令来安装它们：

```shell
pip install -r src/requirements.txt
```

## 实验方法

项目中使用的方法可以在各个 Jupyter notebooks 中查看。

### 特征提取和数据集创建

* 在 01_dataset.ipynb 中，使用 Librosa 库提取音频特征并进行缩放。
* 在 02_dataset_extended_cnn.ipynb 中，提取了更多特征并使用 PCA 进行特征选择，以减少数据集的维度。此外，还从数据集中提取音频的图像，以便后续用于训练 CNN。

### 训练集的交叉验证

为了确定最适合该项目的训练集，对初始、缩放、扩展和 PCA 处理的数据集进行了交叉验证。结果可以在 03_cross_validation_mlp.ipynb 中查看。

### 超参数调优

在交叉验证结果中选择最佳数据集后，进行了随机搜索以优化网络的超参数。有关结果的详细信息以及测试集评估可在 04_hyperparameter_tuning_mlp.ipynb 中找到。

### CNN 训练和调优

05_cnn.ipynb 展示了在 02_dataset_extended_cnn.ipynb 中得到的图像数据集上使用卷积神经网络的结果。

## 注意

出于性能原因，该 notebook 是在 Google Colab 上执行的。

