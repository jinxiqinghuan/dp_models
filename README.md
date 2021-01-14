# Hello there 
The project implement some famous deep learning models

模型实现文件在models中，大部分是参考其他代码复现，因此不保证和原始模型的完全的一致性。

## 00_EDA.ipynb
Exploratory Data Analysis 探索性数据分析
探索一些数据集的读取和结构


## 01_LeNet_cifar10.py
使用LeNet分类cifar10数据集

## 02_Titannic.ipynb
泰坦尼克号生存分析，包含对数据集的详细EDA和清晰绘图。

## 03_pre_trained_vgg19_cifar10.ipynb
探索torchvision库中预训练vgg19的使用方法，使用特征模块探究图片的风格和内容，最后使用修改后的整个vgg19网络实现cifar10分类

## 04_all_VGG_cifar10.py
本文件是VGG11, VGG11_bn, VGG_13, VGG13_bn, VGG_16, VGG_16_bn, VGG_19, VGG_19_bn的实现，只需要指定对应模型名称即可调用对应模型。参考： [链接一](https://github.com/chengyangfu/pytorch-vgg-cifar10) [链接二](https://github.com/pytorch/vision.git)