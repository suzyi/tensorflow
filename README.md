Something confused me
+ **tensorflow中估计导数的方法什么？**
+ **multiGPU**. How to use multiGPU to accelerate computation?
+ **云计算服务器**. 怎么租用阿里云服务器，或者amazon, tencent, nvidia, Google, facebook等的服务器进行高性能计算？
+ **adam algorithm**. Read the original paper about adam algorithm.
+ **dropout algorithm in tensorflow**. Read the original code for dropout algorithm in tensorflow.
+ Whether NIST is a binary?
+ Tensorflow的图结构是什么? 是表示tensor在图中流过吗?
+ 蓝牙通讯的原理是什么?
# 0 - tensorflow introduction
+ Tensorflow依赖于一个高效的C++后端来进行计算。与后端的这个连接叫做session。一般而言，使用TensorFlow程序的流程是先创建一个图，然后在session中启动它。
# 1 - Installating tensorflow-gpu in ubuntu 16.04 x86-64
## 1 - 0 - **[Official installation document](https://www.tensorflow.org/install/install_linux) on ubuntu**.
## 1 - 1 - CUDA installation
### intro to cuda and install method
There are two methods to install CUDA, including distribution-specific packages (RPM and Deb packages, i.e. a ".deb" file, with install command like "sudo apt-get install cuda") and distribution-independent package (runfile package, i.e. a ".run" file, with a install command maybe `sudo sh cuda_<version>_linux.run` under certain enviroment check action.). The method runfile installation means to install CUDA with a standalone installer which is a ".run" file and is completely self-contained.
+ **uninstall cuda Toolkit and driver**. A cuda Toolkit contains cuda driver, samples source code and other resources. Before installing CUDA, uninstall previously installation that could conflict. Even though you hadn't previously have the cuda installed in your system, you can perform the uninstall step and this will not affect your system. By default, cuda was installed in the directory `/usr/local/cuda-X.Y/`. To uninstall it, execute `sudo /usr/local/cuda-X.Y/bin/uninstall_cuda_X.Y.pl` and then uninstall a Driver runfile installation by `sudo /usr/bin/nvidia-uninstall`. Uninstall RPM/Deb installation by `sudo apt-get --purge remove <package-name>`.
+ **Determine which method would you prefer to install tensorflow, package manager based or runfile based?**. The official document recommend the former, if possible. One shall note that the runfile-based installation require user to disable the Nouveau driver and then do some settings under text mode.
+ **determine matched versions for CUDA Toolkit & cuDNN & tensorflow**. There are strict version matching requirement, e.g cuDNN v7.0. must match with CUDA Toolkit 9.0. The official tensorflow installation document listed some recommanded pairs.
+ **Download [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) & [cuDNN](https://developer.nvidia.com/cudnn)**. CUDA Toolkit package has around 1.4GB, and cuDNN has about three files around 200MB need to be downloaded. By the way, you need register first and answer a questionaire before you are allowed to download cuDNN package.
+ **Install CUDA Toolkit**. After finishing the installation of CUDA Toolkit, you will have CUDA Toolkit, Driver and cuda itself in your system. To see their version, execute `nvcc -V` for Toolkit and `cat /proc/driver/nvidia/version` for cuda Driver. There are some mandatory actions you must do and some optional actions you can ignore it. But we highly recommend you to test on the optional NVIDIA_CUDA-X.Y_Samples.
+ **Perform some mandatory actions**. Some actions must be taken before CUDA Toolkit and driver can be used, see [NVIDIA CUDA INSTALLATION GUIDE FOR LINUX](https://docs.nvidia.com/cuda/) for more details.
## errors & debugging
### After correctly installing tensorflow-gpu a few days, I found the screen resolution is extremely bad and then run `nvidia-smi` in terminal and it shows "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running."
#### troubleshooting
+ Cuda error: by running `cat /proc/driver/nvidia/version`, I found my previously and correctly installed driver, now doesn't exist or more exactly, couldn't be loaded, since the directory `/proc/driver/nvidia` doesn't exits.
+ cuDNN error: Verifying cuDNN by executing `cd cudnn_samples_v7/mnistCUDNN/` --> `make clean && make` --> `./mnistCUDNN`, it output unknown error. However, the first time I check the validation of cuDNN with these same steps, it shows "Test passed!".
+ tensorflow error: When execute the example "Hello, world!" in terminal, it failed to call to cuInit: CUDA_ERROR_UNKNOWN.
#### My solution
+ **uninstall CUDA Toolkit and reinstall**. You just need to uninstall CUDA Toolkit, do not uninstall cuDNN and tensorflow. With this step finished, reboot and you will see a high resolution in the screen.
+ **test on cuDNN to see if it works well now**. Just do the same as the above troubleshooting for cuDNN. If it works correctly, you have got out of this problem and then ignore the following step. But you screen might tell you there are some "fatal error". With this situation, do the next.
+ **uninstall cuDNN**.
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
+ **mnist.train.next_batch [(nootbook)](https://github.com/suzyi/tensorflow/blob/master/tf/next_batch.ipynb)**. Return the next `batch_size` examples from this data set.
+ **3-D tensor, tf.constant, tf.matmul, tf.add [(nootbook)](https://github.com/suzyi/tensorflow/blob/master/tf/matmul.ipynb)**. This notebook contains examples like creating 2-D and 3-D tensors and operations like `tf.constant`, `tf.matmul`, `tf.add`. For tensor A has size of h-by-m-by-n and tensor B has size of h-by-n-by-p, then `tf.matmul(A,B)` gives a tensor with size of h-by-m-by-p. 
# 3 - python operation command
+ **basic terminal command**. Enter the python environment in terminal with command `$ python` or `$ python2`, and back with `$ quit()`.
