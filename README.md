Something confused me
+ **Unable to visit tensorflow.org?** See [(github)](https://github.com/tensorflow/tensorflow/issues/3834) for solution, or simply connect to `tensorflow.google.cn`
+ **All API can and their definition can be found in [tensorflow API web](https://tensorflow.google.cn/api_docs/python/tf/metrics/accuracy)**
+ **A Chinese intro to `datasets API` in Iris dataset can be found in [(sohu)](http://www.sohu.com/a/191717118_390227)**
+ **tensorflow中估计导数的方法什么？** [(source code)](https://github.com/tensorflow/tensorflow/tree/7ad74a0d66c5b8547382dfd3aad503288f051ae9/tensorflow/python/training) [(official intro)](https://www.tensorflow.org/api_guides/python/train) [(recommended reading)](http://ruder.io/optimizing-gradient-descent/)
+ **multiGPU**. How to use multiGPU to accelerate computation?
+ **云计算服务器**. 怎么租用阿里云服务器，或者amazon, tencent, nvidia, Google, facebook等的服务器进行高性能计算？
+ **CNN**. An [great illustration](https://zhuanlan.zhihu.com/p/25249694) of CNN, written by Juefei Zhang, an Alibaba engineerer. For each single command to build a CNN in tensorflow, see [Web(API in jianshu, in Chinese)](https://www.jianshu.com/p/e3a79eac554f) and [Web(github, in English)](https://github.com/jikexueyuanwiki/tensorflow-zh/blob/master/SOURCE/api_docs/python/nn.md).
+ **How to calculate padding and output shape when use `tf.nn.conv2d()` to MNIST?** See [(Web1)](https://blog.csdn.net/jk981811667/article/details/78892480) and [(Web2)](http://cs231n.github.io/convolutional-networks/)
+ **adam algorithm**. Read the original paper about adam algorithm. [Train a filter to acheive a goal, like classification or shape detection, 训练CNN的意义是在训练滤波器，是滤波器对特定的模式有较高的激活](https://www.zhihu.com/question/39022858).
+ **dropout algorithm in tensorflow**. Read the original code for dropout algorithm in tensorflow.
+ Whether NIST is a binary?
+ Tensorflow的图结构是什么? 是表示tensor在图中流过吗?[(understand computational graph)](http://www.bubuko.com/infodetail-2280472.html) 也许这样的方式就是可微分编程！
+ 蓝牙通讯的原理是什么?怎样搭建局域网？
+ Which company choose tensorflow as their development frame? mi.com, jd.com, uber, google, Zhongxing (ZTE) see [(tensorflow)](https://www.tensorflow.org/)
# 0 - tensorflow introduction
+ Tensorflow依赖于一个高效的C++后端来进行计算。与后端的这个连接叫做session。一般而言，使用TensorFlow程序的流程是先创建一个图，然后在session中启动它。
# 1 - Installating tensorflow-gpu in ubuntu 16.04 x86-64
## 1 - 0 - **[Official installation document](https://www.tensorflow.org/install/install_linux) on ubuntu**.
## 1 - 1 - CUDA installation
### intro to cuda and install method
There are two methods to install CUDA, including distribution-specific packages (RPM and Deb packages, i.e. a ".deb" file, with install command like "sudo apt-get install cuda") and distribution-independent package (runfile package, i.e. a ".run" file, with a install command maybe `sudo sh cuda_<version>_linux.run` under certain enviroment check action.). The method runfile installation means to install CUDA with a standalone installer which is a ".run" file and is completely self-contained.
+ **uninstall cuda Toolkit and driver**. A cuda Toolkit contains cuda driver, samples source code and other resources. Before installing CUDA, uninstall previously installation that could conflict. Even though you hadn't previously have the cuda installed in your system, you can perform the uninstall step and this will not affect your system. By default, cuda was installed in the directory `/usr/local/cuda-X.Y/`. To uninstall it, execute `sudo /usr/local/cuda-X.Y/bin/uninstall_cuda_X.Y.pl` and then uninstall a Driver runfile installation by `sudo /usr/bin/nvidia-uninstall`. Uninstall RPM/Deb installation by `sudo apt-get --purge remove <package-name>`.
+ **tf.metrics.accuracy() [(notebook)](https://github.com/suzyi/tensorflow/blob/master/tf/metrics_accuracy.ipynb)**.
+ **Determine which method would you prefer to install tensorflow, package manager based or runfile based?**. The official document recommend the former, if possible. One shall note that the runfile-based installation require user to disable the Nouveau driver and then do some settings under text mode.
+ **determine matched versions for CUDA Toolkit & cuDNN & tensorflow**. There are strict version matching requirement, e.g cuDNN v7.0. must match with CUDA Toolkit 9.0. The official tensorflow installation document listed some recommanded pairs.
+ **Download [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) & [cuDNN](https://developer.nvidia.com/cudnn)**. CUDA Toolkit package has around 1.4GB, and cuDNN has about three files around 200MB need to be downloaded. By the way, you need register first and answer a questionaire before you are allowed to download cuDNN package.
+ **Install CUDA Toolkit**. After finishing the installation of CUDA Toolkit, you will have CUDA Toolkit, Driver and cuda itself in your system. To see their version, execute `nvcc -V` for Toolkit and `cat /proc/driver/nvidia/version` for cuda Driver. There are some mandatory actions you must do and some optional actions you can ignore it. But we highly recommend you to test on the optional NVIDIA_CUDA-X.Y_Samples, especially on "bandwidth" and "deviceQuery" with test method `$ ~/NVIDIA_CUDA-9.0_Samples/1_Utilities/deviceQuery/deviceQuery` and `~/NVIDIA_CUDA-9.0_Samples/1_Utilities/bandwidthTest/bandwidthTest`, until you see Result=pass you can ensure you have CUDA Toolkit correctly installed in your computer. Maybe you will see make error when you execute this command `io@msi:~/NVIDIA_CUDA-9.0_Samples$ make`, just ignore it even though it is provided by CUDA official installation guide.
+ **install cuDNN**. There are tow kind of method to install cuDNN, including "installing from a Tar file" and "Installing from a Debian file". However, I recommend "installing from a Tar file", since you will know this installing method is actually some copy and chmod operations and you will clearly know the target path, which will help the "uninstall cuDNN" if you want.
+ **Perform some mandatory actions**. Some actions must be taken before CUDA Toolkit and driver can be used, see [NVIDIA CUDA INSTALLATION GUIDE FOR LINUX](https://docs.nvidia.com/cuda/) for more details.
## errors & debugging
### After correctly installing tensorflow-gpu a few days, I found the screen resolution is extremely bad and then run `nvidia-smi` in terminal and it shows "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running."
#### troubleshooting
+ Cuda error: by running `cat /proc/driver/nvidia/version`, I found my previously and correctly installed driver, now doesn't exist or more exactly, couldn't be loaded, since the directory `/proc/driver/nvidia` doesn't exits.
+ cuDNN error: Maybe the official provided examples and operation "~/cudnn_samples_v7/mnistCUDNN$ make" & "~/cudnn_samples_v7/mnistCUDNN$ make" couldn't be executed and tell you there exist "fatal eror: driver_types.h: No such files", you can igore it. Even though, there are another method available you can try to test if you had cuDNN right installed in the system- just try the tensorflow "Hello, world!" examples in the terminal to check is there exist errors.
+ tensorflow error: When execute the example "Hello, world!" in terminal, it failed to call to cuInit: CUDA_ERROR_UNKNOWN.
#### My solution
+ **uninstall CUDA Toolkit and reinstall**. You just need to uninstall CUDA Toolkit, do not uninstall cuDNN and tensorflow. With this step finished, reboot and you will see a high resolution in the screen.
+ **test on cuDNN to see if it works well now**. Just do the same as the above troubleshooting for cuDNN. If it works correctly, you have got out of this problem and then ignore the following step. But you screen might tell you there are some "fatal error". With this situation, try tensorflow "Hello, world!" examples in terminal (command line), to see if it's ok. If this example works fine, you can ignore the "make error".
+ **uninstall cuDNN**. Via the Tar file based installing instruction, you can see the installation of cuDNN is actually a copy process thus then you can uninstall cuDNN bu delete these files- if you want.
# 2 - Examples of tensorflow
## 2-0 - **References**
+ **[api_docs](https://tensorflow.google.cn/api_docs/python/)**. The official api_doc provides an explicit illustration for each single command.
+ **Examples can be found in [here](https://github.com/suzyi/TensorFlow-Examples)**. There are many examples including basic [Hello, world!](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/1_Introduction/helloworld.ipynb) and advanced operation such as [CNN](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb).
+ Now verifying the cuDNN: change to the directory `cd ~/cudnn_samples_v7/mnistCUDNN/`, `make clean && make`, and execute `./mnistCUDNN`, then output unknown error, which mean cuDNN doesn't work properly. However, the first time I do the same steps it output "Test passed!".
+ By checking tensorflow with example "Hello, world!" in the terminal, it failed call to cuInit: CUDA_ERROR_UNKNOWN. However, this example can be executed in Jupyter-notebook, maybe it doesn't use the GPU.

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
+ **Read MNIST using tensorflow in python ([nootbook](https://github.com/suzyi/tensorflow/blob/master/tf/readMNIST.ipynb))**. This nootbook contains codes for reading the images and labels in MNIST. `one_hot=True` gives label with the vector form that has only one non-zero element with value 1 in the vector, like `array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])` represents 4, and `one_hot=False` directly gives 4.
+ **batch_x, batch_y = mnist.train.next_batch(batch_size)**. `batch_x` is of type `numpy.ndarray` with size 'batch_size-by-784', note that `28*28=784` represent any figure in mnist has 784 pixels in tatal. `print(batch_x)`, `print(type(batch_x))` and `print(np.shape(batch_x)` are all available for knowing the attributes of `batch_x`, and use `print(batch_x[:,0])` to see its first column.
## 2-2 - Understanding of some single commands
+ **tf.data.Dataset.from_tensor_slices(), dataset.batch(batch_size) [(notebook)](https://github.com/suzyi/tensorflow/blob/master/tf/from_tensor_slices.ipynb)**
+ **tf.placeholder(dtype, shape=none, name=none), feed_dict ([nootbook](https://github.com/suzyi/tensorflow/blob/master/tf/feed_dict.ipynb))**. By default, `shape=none` represents a univariate variable. `shape=[None, 3]`, gives a matrix with elements are variables, and it has size of 3 columns but under-determined rows.
+ **tf.nn.dropout [(notebook)](https://github.com/suzyi/tensorflow/blob/master/tf/dropout.ipynb)**. This command haven't been completely figured out.
+ **batch,[将数据集按mini_batch划分](https://sthsf.github.io/wiki/Algorithm/DeepLearning/Tensorflow%E5%AD%A6%E4%B9%A0%E7%AC%94%E8%AE%B0/Tensorflow%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86---%E8%AE%AD%E7%BB%83%E6%A0%B7%E6%9C%AC%E7%9A%84batch_size%E6%95%B0%E6%8D%AE%E7%9A%84%E5%87%86%E5%A4%87.html)**
深度学习的优化算法，说白了就是梯度下降，每次的参数更新有两种方式:
第一种，遍历全部数据集算一次损失函数，然后算函数对各个参数的梯度，更新梯度。这种方法每更新一次参数都要把数据集里的所有样本都看一遍，计算量开销大，计算速度慢，不支持在线学习，这称为Batch gradient descent，批梯度下降。
另一种，每看一个数据就算一下损失函数，然后求梯度更新参数，这个称为随机梯度下降，stochastic gradient descent。这个方法速度比较快，但是收敛性能不太好，可能在最优点附近晃来晃去，hit不到最优点。两次参数的更新也有可能互相抵消掉，造成目标函数震荡的比较剧烈。
为了克服两种方法的缺点，现在一般采用的是一种折中手段，mini-batch gradient decent，小批的梯度下降，这种方法把数据分为若干个批，按批来更新参数，这样，一个批中的一组数据共同决定了本次梯度的方向，下降起来就不容易跑偏，减少了随机性。另一方面因为批的样本数与整个数据集相比小了很多，计算量也不是很大。
基本上现在的梯度下降都是基于mini-batch的，所以深度学习框架的函数中经常会出现batch_size，就是指这个。 
+ **[epoch & iteration](http://blog.csdn.net/u013041398/article/details/72841854)**. 举例说明：训练集有1000个样本，batchsize=10，那么，
训练整个样本集需要：100次iteration，1次epoch。1次epoch表示每个样本只用一次。具体的计算公式为：one epoch = numbers of iterations = N = 样本的数量/batch_size
+ **[layer = tf.layer.dense(inputs,units,activation=none,use_bias=true)](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/layers/core.py)**. Each row of `inputs` is as a single example. `units` is the dimensionality of output space. Build a densely connected network. Default activation is linear function. Default use a bias. The output `layer` has columns of `units`.
+ **class [tf.Graph](http://wiki.jikexueyuan.com/project/tensorflow-zh/api_docs/python/framework.html#Graph)**. A TensorFlow computation, represented as a dataflow graph. A Graph contains a set of Operation objects, which represent units of computation; and Tensor objects, which represent the units of data that flow between operations.
+ **[tf.nn.softmax_cross_entropy_with_logits](http://blog.csdn.net/mao_xiao_feng/article/details/53382790)  [(nootbook)](https://github.com/suzyi/tensorflow/blob/master/tf/softmax_cross_entropy_with_logits.ipynb)**. See wikipedia for [cross entropy](https://en.wikipedia.org/wiki/Cross_entropy).
+ **tf.nn.sparse_softmax_cross_entropy_with_logits [(nootbook)](https://github.com/suzyi/tensorflow/blob/master/tf/sparse_softmax_cross_entropy_with_logits.ipynb)**.
+ **[tf.nn.con2d](https://github.com/HawKsword/deep_learning-machine_learning/blob/83e29893073e297ea5c45622859c1c477f198d33/cnn_tensorflow)**.
+ **tf.argmax [(nootbook)](https://github.com/suzyi/tensorflow/blob/master/tf/argmax.ipynb)**
+ **tf.cast [(nootbook)](https://github.com/suzyi/tensorflow/blob/master/tf/cast.ipynb)**
+ **tf.equal [(nootbook)](https://github.com/suzyi/tensorflow/blob/master/tf/equal.ipynb)**
+ **tf.reduce_mean [(nootbook)](https://github.com/suzyi/tensorflow/blob/master/tf/reduce_mean.ipynb)**
+ **tf.random_uniform(shape,minval=0,maxval=None,dtype=tf.float32,seed=None,name=None)**. Return a uniformly distributed random values with shape are nessary to provide.
+ **tf.random_normal [(nootbook)](https://github.com/suzyi/tensorflow/blob/master/tf/random_normal.ipynb)**. Create random number, seed is optional.
+ **tensorboard [(nootbook)](https://github.com/suzyi/tensorflow/blob/master/graph/tensorboard.ipynb)**. Graph visualisation.
+ **tf.Variable [(nootbook)](https://github.com/suzyi/tensorflow/blob/master/tf/Variable.ipynb)**. Usage of tf.Variable().
+ **tf.Session [(nootbook)](https://github.com/suzyi/tensorflow/blob/master/tf/Session.ipynb)**.
+ **mnist.train.next_batch [(nootbook)](https://github.com/suzyi/tensorflow/blob/master/tf/next_batch.ipynb)**. Return the next `batch_size` examples from this data set.
+ **3-D tensor, tf.constant, tf.matmul, tf.add [(nootbook)](https://github.com/suzyi/tensorflow/blob/master/tf/matmul.ipynb)**. This notebook contains examples like creating 2-D and 3-D tensors and operations like `tf.constant`, `tf.matmul`, `tf.add`. For tensor A has size of h-by-m-by-n and tensor B has size of h-by-n-by-p, then `tf.matmul(A,B)` gives a tensor with size of h-by-m-by-p. 
+ **tf.nn.bias_add(), tf.add() [(notebook)](https://github.com/suzyi/tensorflow/blob/master/tf/bias_add.ipynb)**. We show the difference betweent the `tf.nn.bias_add()` and `tf.add()`.
### 2-2-2 - some single commands in building CNN-type network
+ **`weight={'wcl': tf.random_normal([2,3,1,7])}` [(notebook)]()**
# 3 - python operation command
+ **basic terminal command**. Enter the python environment in terminal with command `$ python` or `$ python2`, and back with `$ quit()`.
