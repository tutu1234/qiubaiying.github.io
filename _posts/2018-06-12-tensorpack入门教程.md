---
layout:     post
title:      tensorpack入门教程
subtitle:   推荐使用tensorpack作为深度学习研究和工程框架
date:       2018-06-12
author:     Hututu
header-img: img/post-bg-2015.jpg
catalog: true
tags:
    - Deep Learning
---

### 前言
- [tensorpack的github地址](https://github.com/tensorpack/tensorpack#toc5)
- 在实验室和近期的实习都会用深度学习的方法解决问题，大家基本上会复现论文的方法完成工作，大部分开源工程都是基于`tensorflow`来实现，而`low-level`的`tensorflow api`用起来很麻烦，并且神经网络的训练和测试过程包括了数据加载，模型选择，日志记录，训练策略等问题，特别是对于大数据，如何提高数据加载速度，进而提高`GPU`利用率和训练速度，对于训练过程至关重要。
- `tensorpack`的`dataflow`模块使用多进程和多线程加载数据，使用`CPU`和`GPU`队列，隐藏了数据转移到内存和显存的延迟，十分高效，同时提供了很多公开数据集的读取接口，方便实验，包括`ImageNet， mnist， cifar, shvn, bsds500`数据集，`dataflow`可以脱离`tensorpack`使用，[dataflow使用范例](#jumpa)
- `tensorpack`的`callbacks`提供除了迭代训练`(minimize train_loss)`之外的接口，除了迭代训练之外，还需要：
	- 在训练开始之前（初始化`tf.session, tf.saver`，转存`graph`）
	- 伴随每一次迭代（`graph`中的操作）
	- 每次迭代之间   （更新超参数，进度条）
	- 每次`epoch`之间（保存模型，验证集）
	- 训练完成之后（提示等）
  
  传统的方式是把额外的操作和训练迭代写到一起，代码冗长，通过写`callback`实现共有的额外操作，`tensorpack trainer`会在恰当的时间点调用，因此共用的额外操作只要`tensorpack trainer`使用一行代码即可。
- `tensorpack`的`trainer`包含的逻辑是：
	- Building the graph
	- Running the iterations(with callbacks)
 
  通常不直接使用这些方法，而是使用`high-level interface`，只需要选择使用哪一个`trainer`，但是关于`trainer`的基本知识，我们还是要了解。
- `tensorpack`更多的方便功能，包括：
	- 数据输入的`Pipeline`
	- 使用封装的`Symbolic layers`定义`graph`
	- 加载和保存模型(`.npy, .npz, .checkpoint`)
	- 总结和日志等

###   <span id = "jumpa">dataflow使用范例</span>
1. 参考[Effcient Dataflow](http://tensorpack.readthedocs.io/tutorial/efficient-dataflow.html)
2. 写自己的dataflow，参考[Write a DataFlow](http://tensorpack.readthedocs.io/tutorial/extend/dataflow.html)

### callback使代码无比简洁


### Trainer一行代码设置


### 更多的功能
1. `tensorpack`中还有很多优秀的功能，大家可以拿过来直接用，比如数据预处理，模型定义，日志记录等，太多了，大家快快阅读`tensorpack`的[文档](http://tensorpack.readthedocs.io/tutorial/index.html#)和[源码](https://github.com/tensorpack/tensorpack#toc5)，你一定会惊叹它是多么方便。
2. 最后，吹一波`tensorpack`的作者吴育昕大神，[知乎](https://www.zhihu.com/people/ppwwyyxx/activities),[博客](http://ppwwyyxx.com/),去膜拜吧！
