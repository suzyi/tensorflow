# 0 - Something confused me
+ **tensorflow中估计导数的方法什么？**
+ **multiGPU**. How to use multiGPU to accelerate computation?
+ **云计算服务器**. 怎么租用阿里云服务器，或者amazon, tencent, nvidia, Google, facebook等的服务器进行高性能计算？
+ **adam algorithm**. Read the original paper about adam algorithm.
+ **dropout algorithm in tensorflow**. Read the original code for dropout algorithm in tensorflow.
+ Whether NIST is a binary?
+ Tensorflow的图结构是什么? 是表示tensor在图中流过吗?
# 1 - tensorflow installation
**[Official installation document](https://www.tensorflow.org/install/install_linux) on ubuntu**.
# 2 - Examples of tensorflow
## 2-0 - **References**
+ **[api_docs](https://tensorflow.google.cn/api_docs/python/)**. The official api_doc provides an explicit illustration for each single command.
+ **Examples can be found in [here](https://github.com/suzyi/TensorFlow-Examples)**. There are many examples including basic [Hello, world!](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/1_Introduction/helloworld.ipynb) and advanced operation such as [CNN](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb).


## 2-1 - **understand [mnist](http://yann.lecun.com/exdb/mnist/)**.
### terminology
+ **Handwritten digits with values from 0 to 1**. 我的理解：每张图片上的数字与背景都会有色差，把背景色定义为0，色差最深（绝对值最大）的点定义为1，其他色差按比例定义一个(0,1)区间上的浮点数，比如0.4, 0.77等等，然后就可以将一幅图片转换为一个矩阵。
+ **label**. 标签如果是3，那么标签是向量(0，0，0，1，0，0，0，0，0，0)，除了第4个值为1，其他全为0
+ **[binary image](https://en.wikipedia.org/wiki/Binary_image)**. A binary image is a digital image that has only two possible values for each pixel. Typically, the two colors used for a binary image are black and white. The color used for the object(s) in the image is the foreground color while the rest of the image is the background color. Binary images are also called bi-level or two-level. This means that each pixel is stored as a single bit—i.e., a 0 or 1. The names black-and-white, B&W, monochrome or monochromatic are often used for this concept.
+ **[grey level image](https://en.wikipedia.org/wiki/Grayscale)**. Grayscale images are distinct from one-bit bi-tonal black-and-white images, which in the context of computer imaging are images with only two colors, black and white (also called bilevel or binary images). Grayscale images have many shades of gray in between.
+ **[aspect ratio](https://en.wikipedia.org/wiki/Aspect_ratio_(image))**. For example, a group of images that all have an aspect ratio of 16:9, one image might be 16 inches wide and 9 inches high, another 16 centimeters wide and 9 centimeters high, and a third might be 8 yards wide and 4.5 yards high. Thus, aspect ratio concerns the relationship of the width to the height, not an image's actual size.
### understanding of MNIST
+ **[NIST](https://www.nist.gov/srd/nist-special-database-19)**. NIST is an original database containing many (binary, at least SD-3 and SD-1 are.) images with handprinting digits or aplabetic characters.
+ **MNIST**. MNIST is a grey levels image database constructed from NIST's Special Database 3 (SD-3) and Special Database 1 (SD-1) with the anti-aliasing technique. One should note that both SD-3 and SD-1 contain binary images. MNIST was normalized to fit a 28-by-28 pixel box while preserving ratio. It has 60000 handwitting images for training and 60000 for testing, while only a subset of 10000 test images is available. The full 60000 training set is available. All files in MIST are not in any standard image format so you have to write your own program to read them.
+ **Read MNIST using tensorflow in python ([nootbook](https://github.com/suzyi/tensorflow/blob/master/readMNIST.ipynb))**. This nootbook contains codes for reading the images and labels in MNIST.
## 2-2 - Understanding of some single commands
+ **a=placeholder(tf.int16)**. 
+ **batch,[将数据集按mini_batch划分](https://sthsf.github.io/wiki/Algorithm/DeepLearning/Tensorflow%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/Tensorflow%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86---%E8%AE%AD%E7%BB%83%E6%A0%B7%E6%9C%AC%E7%9A%84batch_size%E6%95%B0%E6%8D%AE%E7%9A%84%E5%87%86%E5%A4%87.html)**
深度学习的优化算法，说白了就是梯度下降，每次的参数更新有两种方式:
第一种，遍历全部数据集算一次损失函数，然后算函数对各个参数的梯度，更新梯度。这种方法每更新一次参数都要把数据集里的所有样本都看一遍，计算量开销大，计算速度慢，不支持在线学习，这称为Batch gradient descent，批梯度下降。
另一种，每看一个数据就算一下损失函数，然后求梯度更新参数，这个称为随机梯度下降，stochastic gradient descent。这个方法速度比较快，但是收敛性能不太好，可能在最优点附近晃来晃去，hit不到最优点。两次参数的更新也有可能互相抵消掉，造成目标函数震荡的比较剧烈。
为了克服两种方法的缺点，现在一般采用的是一种折中手段，mini-batch gradient decent，小批的梯度下降，这种方法把数据分为若干个批，按批来更新参数，这样，一个批中的一组数据共同决定了本次梯度的方向，下降起来就不容易跑偏，减少了随机性。另一方面因为批的样本数与整个数据集相比小了很多，计算量也不是很大。
基本上现在的梯度下降都是基于mini-batch的，所以深度学习框架的函数中经常会出现batch_size，就是指这个。 
+ **[epoch & iteration](http://blog.csdn.net/u013041398/article/details/72841854)**. 举例说明：训练集有1000个样本，batchsize=10，那么，
训练整个样本集需要：100次iteration，1次epoch。1次epoch表示每个样本只用一次。具体的计算公式为：one epoch = numbers of iterations = N = 样本的数量/batch_size
+ **class [tf.Graph](http://wiki.jikexueyuan.com/project/tensorflow-zh/api_docs/python/framework.html#Graph)**. A TensorFlow computation, represented as a dataflow graph. A Graph contains a set of Operation objects, which represent units of computation; and Tensor objects, which represent the units of data that flow between operations.
+ **[tf.nn.softmax_cross_entropy_with_logits](http://blog.csdn.net/mao_xiao_feng/article/details/53382790)**. See wikipedia for [cross entropy](https://en.wikipedia.org/wiki/Cross_entropy).
+ **tf.reduce_mean [(nootbook)](http://blog.csdn.net/qq_32166627/article/details/52734387)**
+ **tf.random_normal [(nootbook)](https://github.com/suzyi/tensorflow/blob/master/random_normal.ipynb)**. Create random number, seed is optional.
+ **tf.Variable [(nootbook)](https://github.com/suzyi/tensorflow/blob/master/Variable.ipynb)**. Usage of tf.Variable().
+ **tf.Session [(nootbook)](https://github.com/suzyi/tensorflow/blob/master/Session.ipynb)**.
+ **3-D tensor, tf.constant, tf.matmul, tf.add [(nootbook)](https://github.com/suzyi/tensorflow/blob/master/tf/matmul.ipynb)**. This notebook contains examples like creating 2-D and 3-D tensors and operations like `tf.constant`, `tf.matmul`, `tf.add`. For tensor A has size of h-by-m-by-n and tensor B has size of h-by-n-by-p, then `tf.matmul(A,B)` gives a tensor with size of h-by-m-by-p. 
# 3 - python operation command
+ **basic terminal command**. Enter the python environment in terminal with command `$ python` or `$ python2`, and back with `$ quit()`.
