# 1 - tensorflow installation
**[Official installation document](https://www.tensorflow.org/install/install_linux) on ubuntu**.
# 2 - Examples of tensorflow
## References
+ **[api_docs](https://tensorflow.google.cn/api_docs/python/)**. The official api_doc provides an explicit illustration for each single command.
+ **Examples can be found in [here](https://github.com/suzyi/TensorFlow-Examples)**. There are many examples including basic [Hello, world!](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/1_Introduction/helloworld.ipynb) and advanced operation such as [CNN](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb).

## Understanding of some single commands
+ **a=placeholder(tf.int16)**. 
+ **batch,[将数据集按mini_batch划分](https://sthsf.github.io/wiki/)**
深度学习的优化算法，说白了就是梯度下降，每次的参数更新有两种方式:
第一种，遍历全部数据集算一次损失函数，然后算函数对各个参数的梯度，更新梯度。这种方法每更新一次参数都要把数据集里的所有样本都看一遍，计算量开销大，计算速度慢，不支持在线学习，这称为Batch gradient descent，批梯度下降。
另一种，每看一个数据就算一下损失函数，然后求梯度更新参数，这个称为随机梯度下降，stochastic gradient descent。这个方法速度比较快，但是收敛性能不太好，可能在最优点附近晃来晃去，hit不到最优点。两次参数的更新也有可能互相抵消掉，造成目标函数震荡的比较剧烈。
为了克服两种方法的缺点，现在一般采用的是一种折中手段，mini-batch gradient decent，小批的梯度下降，这种方法把数据分为若干个批，按批来更新参数，这样，一个批中的一组数据共同决定了本次梯度的方向，下降起来就不容易跑偏，减少了随机性。另一方面因为批的样本数与整个数据集相比小了很多，计算量也不是很大。
基本上现在的梯度下降都是基于mini-batch的，所以深度学习框架的函数中经常会出现batch_size，就是指这个。 
