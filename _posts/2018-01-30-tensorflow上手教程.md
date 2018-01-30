---
layout:     post
title:      tensorflow上手教程
subtitle:   tensorflow and computional graph
date:       2018-01-30
author:     Hututu
header-img: img/post-bg-2015.jpg
catalog: true
tags:
    - Deep Learning
---
## tensorflow

### 简介（introduction）
`TensorFlow`是`Google`在`2015`年`11`月份开源的人工智能系统[Github项目地址](https://github.com/tensorflow/tensorflow)，官网介绍为：`TensorFlow™ `是一个使用数据流图进行数值计算的开源软件库。图中的节点代表数学运算， 而图中的边则代表在这些节点之间传递的多维数组（张量）。这种灵活的架构可让您使用一个`API`将计算工作部署到桌面设备、服务器或者移动设备中的一个或多个`CPU`或`GPU`。` TensorFlow` 最初是由 `Google` 机器智能研究部门的 `Google Brain` 团队中的研究人员和工程师开发的，用于进行机器学习和深度神经网络研究， 但它是一个非常基础的系统，因此也可以应用于众多其他领域。

### 构建神经网络（Build neural networks）
1. About Tensorflow
	- tensorlow是Google开源的基于python的神经网络工具包
	- 基于数据流图(data-flow graph based),节点(node)代表数学运算，边(edge)代表张量(tensor)
2. 基本原则(General Principle)
	- 使用tf构建NN，基本按照如下步骤，会逐一介绍:
        - import需要的模块.
		```python
        import tensorflow as tf
        import numpy as np
        ```
		- 定义NN所需要的layers.
		- 定义variables并初始化.
		- 创建session来执行运算.
		- 搭建NN,用placeholders和feed_dict喂给NN数据.
		- 训练和测试NN, 用Tensorboard可视化网络并观察结果.
2. Session
	- 为了控制程序的操作，需要创建session，在session中运行图中运算，有两种方式:
	```python
    #create two matrix
    matrix1 = tf.constant([[3,3]])
	matrix2 = tf.constant([[2],[2]])
	product = tf.matmul(matrix1,matrix2)
    #-method 1: create a variable for a session
    sess = tf.Session()
    result = sess.run(product)
    print(result)
    sess.close() #Remember to close the session if using this method!
    #-method 2: use with to create a session
    with tf.Session() as sess:
    	result2 = sess.run(product)
        print(result2)
    ```
3. Variable & Scope
	- 有两种scopes，两种创建variables的方法:
	```python
    #I. tf.name_scope()
    with tf.name_scope('a_name_scope'):
    	initializer = tf.constant_initializer(value=1)
        var1 = tf.get_variable(name='var1',shape=[1],dtype=tf.float32,initializer=initializer)
        var2 = tf.Variable(name='var2',initial_value=[2],dtype=tf.float32)
        var21= tf.Variable(name='var2',initial_value=[2.1],dtype=tf.float32)
        var22= tf.Variable(name='var2',initial_value=[2.2],dtype=tf.float32)
    #Result of 1:
    with tf.Session() as sess:
    	sess.run(tf.initialize_all_variables())
        print(var1.name)
        print(sess.run(var1))		#tf.get_variable→no effect!! tf.Variable→names will change!!
        print(var2.name)
        print(sess.run(var2))
        print(var21.name)
        print(sess.run(var21))
        print(var22.name)
        print(sess.run(var22))
        
        #II. tf.variable_scope()→to reuse the variables
        #If you want to reuse some variables, remember to add the following line:
        scope.reuse_variables()
        with tf.variable_scope("a_variable_scope") as scope:
        	initializer = tf.constant_initializer(value=3)
            var3 = tf.get_variable(name='var3',shape=[1],dtype=tf.float32,initializer=initializer)
            scope.reuse_variables()
            var3_reuse = tf.get_variable(name='var3')
            var4 = tf.Variable(name='var4',initial_value=[4],dtype=tf.float32)
            var4_reuse = tf.Variable(name='var4',initial_value=[4],dtype=tf.float32)
        #Result of 2:
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            print(var3.name)
            print(sess.run(var3))	#tf.get_variable→can be reused!! tf.Variable→names will change!!
            print(var3_reuse.name)
            print(sess.run(var3_reuse))
            print(var4.name)
            print(sess.run(var4))
            print(var4_reuse.name)
            print(sess.run(var4_reuse))
    ```
    - 定义任何variable,牢记用下面的命令初始化(建图过程),随后在session中执行初始化:
    ```python
    #So important that we repeat 3 times !!
    init = tf.global_variables_initializer()
	init = tf.global_variables_initializer()
	init = tf.global_variables_initializer()
    sess.run(init)
    ```
4. Placeholder
	- 将outside的data喂给NN，需要使用placeholder作为一个容器:
	```python
    # Define the data type of the placeholder
    input1 = tf.placeholder(tf.float32)
    input2 = tf.placeholder(tf.float32)
    
    output = tf.multiply(input1,input2)
    ```
    - 使用session执行**喂**这个操作，使用`feed_dict={}`指定喂的变量
    ```python
    with tf.Session() as sess:
    	out = sess.run(output, feed_dict={input1:[7.], input2:[2.]})
        print(out)
    ```
6. Define add_layer
	 - 定义`add_layer`，使得`layer`的添加方便易读
	 ```python
     def add_layer(inputs, in_size, out_size, activate_function=None):
     	Weights = tf.Variable(tf.random_normal([insize, out_size]))
         biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
         Wx_plus_b = tf.matmul(inputs, Weights) + biases
         if activation_function is None:
        	outputs = Wx_plus_b
         else:
        	outputs = activation_function(Wx_plus_b)
        return outputs
     ```
7. Optimizer
	- NN优化问题涉及复杂的优化算法，会单独详解
8. Example: building a simple neural network
	- 大家可以参考[莫烦的例子](https://github.com/MorvanZhou/tutorials/tree/master/tensorflowTUT/tf11_build_network)
	- 实际敲代码才会有跟深刻的理解,上面的例子会帮助你掌握用tensorflow搭建一个简单的NN。

### 可视化(Visulization)
1. matplotlib可视化我们的结果，可以通过Plot的方式：
```python
	#1.import the module
    import matplotlib.pyplot as plt
    #2. plt.figure()→add_subplot()→plt.show()
    #plot the real data
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x_data, y_data) #ax.scatter→scattering plot  ax.line→line
    #3. For continuous display, add the following line:
    plt.ion()
```
2. tensorboard可视化NN
	- Tensorflow features Tensorboard,可以帮助我们分析我们设计的NN
![](http://ww1.sinaimg.cn/large/8833244fly1fnyllh3890j20kq0b3q3n.jpg)
	- 使用介绍,tensorboard的详细我会在另一篇博文中详细介绍
	```python
    #1. Define the names (inputs, weights ,biases, loss, ...),EX:input
    with tf.name_scope("inputs"):
    	xs = tf.placeholder(tf.float32, [None,1], name='x_input')
        ys = tf.placeholder(tf.float32, [None,1], name='y_input')
   #2. Create a session and use tf.summary.FileWriter() to restore the graph
    sess = tf.Session()
    writer = tf.summary.FileWriter('logs/', sess.graph)
    #3. Go back to your terminal and enter the following command line:
    tensorboard --logdir='logs/'
    #4. Copy the website address displayed on your terminal to the web browser and you can see the Tensorboard!
    ```
    - summary图解
    ```
    I. tf.scalar_summary()→loss
	II. tf.histogram_summary()→others(weights, biases, outputs, ...)
    Note: 
(1) These two kinds of summary diagrams must be used together, otherwise there may be some bugs!
(2) The diagrams will dynamically change when the program is running!
    ```

### 附录(Appendix)

- I. 激活函数(Activation functions)
	- https://www.tensorflow.org/versions/r0.10/api_docs/python/nn/activation_functions_
- II. 优化方法(Optimizers)
	-	https://www.tensorflow.org/api_guides/python/train
	-	Basic: GradientDescentOptimizer
	-	Advanced: MomentumOptimizer & AdamOptimizer
	- Alphago (a.k.a. master): RMSPropOptimizer
	 





### 引用
 - I. Morvan Zhou’s youtube channel & website
	- https://www.youtube.com/playlist?list=PLXO45tsB95cKI5AIlf5TxxFPzb-0zeVZ8
	- https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/
- II. Tensorflow
	- https://www.tensorflow.org/
- III. CS224d: Tensorflow Tutorial
	- https://cs224d.stanford.edu/lectures/CS224d-Lecture7.pdf

