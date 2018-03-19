# 0 - Something confused me
+ **tensorflow中估计导数的方法什么？**
+ **multiGPU**. How to use multiGPU to accelerate computation?
+ **云计算服务器**. 怎么租用阿里云服务器，或者amazon, tencent, nvidia, Google, facebook等的服务器进行高性能计算？
# 1 - tensorflow installation
**[Official installation document](https://www.tensorflow.org/install/install_linux) on ubuntu**.
# 2 - Examples of tensorflow
## References
+ **[api_docs](https://tensorflow.google.cn/api_docs/python/)**. The official api_doc provides an explicit illustration for each single command.
+ **Examples can be found in [here](https://github.com/suzyi/TensorFlow-Examples)**. There are many examples including basic [Hello, world!](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/1_Introduction/helloworld.ipynb) and advanced operation such as [CNN](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb).

## Understanding of some single commands
+ **a=placeholder(tf.int16)**. 
+ **batch,[将数据集按mini_batch划分](https://sthsf.github.io/wiki/Algorithm/DeepLearning/Tensorflow%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/Tensorflow%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86---%E8%AE%AD%E7%BB%83%E6%A0%B7%E6%9C%AC%E7%9A%84batch_size%E6%95%B0%E6%8D%AE%E7%9A%84%E5%87%86%E5%A4%87.html)**
深度学习的优化算法，说白了就是梯度下降，每次的参数更新有两种方式:
第一种，遍历全部数据集算一次损失函数，然后算函数对各个参数的梯度，更新梯度。这种方法每更新一次参数都要把数据集里的所有样本都看一遍，计算量开销大，计算速度慢，不支持在线学习，这称为Batch gradient descent，批梯度下降。
另一种，每看一个数据就算一下损失函数，然后求梯度更新参数，这个称为随机梯度下降，stochastic gradient descent。这个方法速度比较快，但是收敛性能不太好，可能在最优点附近晃来晃去，hit不到最优点。两次参数的更新也有可能互相抵消掉，造成目标函数震荡的比较剧烈。
为了克服两种方法的缺点，现在一般采用的是一种折中手段，mini-batch gradient decent，小批的梯度下降，这种方法把数据分为若干个批，按批来更新参数，这样，一个批中的一组数据共同决定了本次梯度的方向，下降起来就不容易跑偏，减少了随机性。另一方面因为批的样本数与整个数据集相比小了很多，计算量也不是很大。
基本上现在的梯度下降都是基于mini-batch的，所以深度学习框架的函数中经常会出现batch_size，就是指这个。 
+ **[epoch & iteration](http://blog.csdn.net/u013041398/article/details/72841854)**. 举例说明：训练集有1000个样本，batchsize=10，那么，
训练整个样本集需要：100次iteration，1次epoch。1次epoch表示每个样本只用一次。具体的计算公式为：one epoch = numbers of iterations = N = 样本的数量/batch_size
+ **mnist**.Handwritten digits with values from 0 to 1. 我的理解：每张图片上的数字与背景都会有色差，把背景色定义为0，色差最深（绝对值最大）的点定义为1，其他色差按比例定义一个(0,1)区间上的浮点数，比如0.4, 0.77等等，然后就可以将一幅图片转换为一个矩阵。
+ **class [tf.Graph](http://wiki.jikexueyuan.com/project/tensorflow-zh/api_docs/python/framework.html#Graph)**. A TensorFlow computation, represented as a dataflow graph. A Graph contains a set of Operation objects, which represent units of computation; and Tensor objects, which represent the units of data that flow between operations.
+ **[tf.nn.softmax_cross_entropy_with_logits](http://blog.csdn.net/mao_xiao_feng/article/details/53382790)**
# 4 - python operation command
+ **basic terminal command**. Enter the python environment in terminal with command `$ python` or `$ python2`, and back with `$ quit()`.
