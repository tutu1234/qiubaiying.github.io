---
layout:     post
title:      DeepLearning实验规范化
subtitle:   规范实验流程，对比分析结果，逐步解决问题
date:       2018-06-28
author:     Hututu
header-img: img/post-bg-2015.jpg
catalog: true
tags:
    - Deep Learning
---
为无为，事无事，味无味，大小多少。报怨以德。
图难于其易，为大于其细;
天下难事，心作于易；天下大事，心作于细。
——老子
### DeepLearning实验规范的关键点
- DeepLearning是一门玄学，往往能得到十分出色的结果，但是达到产品应用的标准，需要大量的实验优化。
- DeepLearning的做产品需要高质量的数据作为训练集，大量的数据覆盖所有情况，同时产品上线需要在公司QA的测试集达到好的结果，测试集不需要大量数据，不能和训练集有重合，但要覆盖所有情况，比如光照、场景、角度等，模型在训练集上刷高是必须的，但可能训练集中某些情况的数据覆盖的少，导致在QA测试集中识别错误，这个时候就需要在训练集中添加对应的样本不断迭代，这也是产品和竞赛的区别，竞赛的数据集是固定的，提高正确率的关键在模型设计和算法。
**注意：更新数据集之后需要重新计算数据集的均值和方差**
- DeepLearning也是一门数据科学，大量的数据需要记录，分析，对比。希望能有一个统一的框架帮助我们，是我们的焦点在模型和算法上，[tensorpack](https://github.com/tensorpack/tensorpack#toc5)是一个优秀的框架，但是需要tensorflow和python语言的基础，可以通过其中的examples逐步学习，tensorpack隐藏了数据加载，日志记录，训练超参数定义等重复的过程，帮我们很容易的进行实验和结果记录。
- 准备好数据和框架，就可以开始在服务器中进行训练实验，需要了解服务器的配置，包括GPU，CPU，Memory，Disk等，配置好的机器能帮我们更快的完成实验，可以通过以下命令查看配置：
```
GPU: nvidia-smi -l 2
CPU/Mem: htop, free, cat /proc/cpuinfo
cat /sys/block/*/queue/rotational    （其中*为你的硬盘设备名称，例如sda等等），如果返回1则表示磁盘可旋转，那么就是HDD了
```
- 准备好了机器，数据和框架，我们可以进行对比实验进行分析优化了，这就要求我们把实验结果规范地记录，什么时间，在什么条件下，做的什么实验，可以先分析不同的实验条件，根据条件不同，列出需要实验的表格，理清思路，比如：
```
实验   sessinit       structure    loss               accu
1     imagenet_res18  resnet18    cross_entropy      0.0     
2     None  		  resnet18    cross_entropy	  0.0
3     imagenet_res18  resnet18    contrasive_loss    0.0
```
不同的实验的配置文件和脚本及产生的数据，分别以不同的实验条件来命名文件夹。文件结构如：
```
antispoof/
    |->experiments
    |    |->Res18_baseline
    		|->*.py
         |->Res18_SimilarityLearning
         	|->*.py
    |->train_log
    |    |->antispoof-resnet-d18-width1-entropyloss
    |    |->antispoof-resnet-d18-width1-tripletloss
    |    |->imagenet_model
    |    |->final_model
    |->data
    |    |->train
    |    |   |->* .txt
    |    |->test
    |    |   |->*.txt
    |    |->log
    |    |->csv
```
- 实验Debug和优化，做一次模型的训练可能就要花费一天的时间，所以尽量不浪费时间就要每次实验都能有所发现，实验之后的实验过程中的参数数据和测试结果都很重要，如何查看分析数据呢？在tensorpack中，对于实验过程中的参数变化和结果，会记录在TFEvent中，通过tensorboard查看，同时日志log和关键的结果json也会记录
- Debug的小提示：
	- tensorboard中查看loss、error、lr的变化
	- 测试集中结果差的数据挑出来
	- 模型结构，损失函数，对比实验
	- 训练集和测试集正确率的变化是否同步

