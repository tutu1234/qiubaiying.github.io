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
为无为，事无事，味无味，大小多少。报怨以德。
图难于其易，为大于其细;
天下难事，心作于易；天下大事，心作于细。
——老子
### 前言
- [tensorpack的github地址](https://github.com/tensorpack/tensorpack#toc5)
- 在实验室和近期的实习都会用深度学习的方法解决问题，大家基本上会复现论文的方法完成工作，大部分开源工程都是基于`tensorflow`来实现，而`low-level`的`tensorflow	api`用起来很麻烦，并且神经网络的训练和测试过程包括了数据加载，模型选择，日志记录，训练策略等问题，特别是对于大数据，如何提高数据加载速度，进而提高`GPU`利用率和训练速度，对于训练过程至关重要。
- 使用`CPU`和`GPU`队列，隐藏了数据转移到内存和显存的延迟，同样加速了训练过程
- `tensorpack`的`dataflow`模块使用多进程和多线程加载数据，十分高效，话不多说，直接上手，以`ImageNet`数据集为例。


### dataflow使用范例
1. 参考[Effcient Dataflow](http://tensorpack.readthedocs.io/tutorial/efficient-dataflow.html)
2. 写自己的dataflow，参考[Write a DataFlow](http://tensorpack.readthedocs.io/tutorial/extend/dataflow.html)


### 更多的功能
1. `tensorpack`中还有很多优秀的功能，大家可以拿过来直接用，比如数据预处理，模型定义，日志记录等，太多了，大家快快阅读`tensorpack`的[文档](http://tensorpack.readthedocs.io/tutorial/index.html#)和[源码](https://github.com/tensorpack/tensorpack#toc5)，你一定会惊叹它是多么方便。
2. 最后，吹一波`tensorpack`的作者吴育昕大神，[知乎](https://www.zhihu.com/people/ppwwyyxx/activities),[博客](http://ppwwyyxx.com/),去膜拜吧！
