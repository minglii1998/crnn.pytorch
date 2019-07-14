Convolutional Recurrent Neural Network
======================================

This software implements the Convolutional Recurrent Neural Network (CRNN) in pytorch.
Origin software could be found in [crnn](https://github.com/bgshih/crnn)

Run demo
--------
A demo program can be found in ``demo.py``. Before running the demo, download a pretrained model
from [Baidu Netdisk](https://pan.baidu.com/s/1pLbeCND) or [Dropbox](https://www.dropbox.com/s/dboqjk20qjkpta3/crnn.pth?dl=0). 
This pretrained model is converted from auther offered one by ``tool``.
Put the downloaded model file ``crnn.pth`` into directory ``data/``. Then launch the demo by:

    python demo.py

The demo reads an example image and recognizes its text content.

Example image:
![Example Image](./data/demo.png)

Expected output:
    loading pretrained model from ./data/crnn.pth
    a-----v--a-i-l-a-bb-l-ee-- => available

Dependence
----------
* [warp_ctc_pytorch](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding)
* lmdb

Train a new model
-----------------
1. Construct dataset following [origin guide](https://github.com/bgshih/crnn#train-a-new-model). If you want to train with variable length images (keep the origin ratio for example), please modify the `tool/create_dataset.py` and sort the image according to the text length.
2. Execute ``python train.py --adadelta --trainRoot {train_path} --valRoot {val_path} --cuda``. Explore ``train.py`` for details.

Additional -Ming
=============

###### 以下部分都由ming后期添加，旨在帮助自己更好的理解这个代码

## 1. 训练

训练要用`create_dataset.py`来制作lmdb格式的数据库。但是由于不知道什么的原因，用自己下载的svt数据库然后转成lmdb时遇到了问题，这个问题暂时还未解决，但是有MJ的lmdb数据库，这个问题暂时不考虑。

在试图训练的过程中，会遇到一些bug，不知道是出于什么样的原因，代码原作者并没有去修改这些bug。我已在`zz bug&handling--Ming.txt`中记录下了这些bug，因此遇到问题直接对照该文档即可，就不再把它搬到readme里了。

## 2. 代码解释

###### 自己只是初学者，只是在自己所理解的范围内稍加解释。

### 2.1 models/crnn.py

在自己的另一个仓库里，有一个[CRNN Learning Note](https://github.com/minglii1998/TextDetection/blob/master/CRNN%20Learning%20Note.md)，可以对照着它来看关于crnn的模型。

#### 2.1.1 class CRNN

这个模型的写法是看着感觉比较高端的写法。在官网的[Training a Classifier](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)的例子中，可以看到模型是很简单的写法，不过..这里就比较复杂了。

```python
Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
```

可以看出，ks ps ss nm 这四个列表分别代表的是kernel_size，padding，stride以及in,out channels。

```python
class Sequential(*args)
A sequential container. Modules will be added to it in the order they are passed in the constructor. Alternatively, an ordered dict of modules can also be passed in.

To make it easier to understand, here is a small example:

    # Example of using Sequential  
    model = nn.Sequential(  
              nn.Conv2d(1,20,5),  
              nn.ReLU(),  
              nn.Conv2d(20,64,5),  
              nn.ReLU()  
            )  
```

上面是Sequential函数的一般用法，但是在该代码中，Sequential后面并没有任何参数，应该和add_module函数有关。[pytorch学习： 构建网络模型的几种方法](https://www.cnblogs.com/denny402/p/7593301.html)这篇博客有介绍这种方法，就是利用Sequential快速搭建，然后为了避免每一层没有名字的不足，就利用add_module函数向其中添加部件，有两个参数，分别是层的名字以及模型。

在添加过程中，pooling部分直接用add_module添加，可是卷积部分并没有直接添加，而是又定义了一个convRelu(i, batchNormalization=False)来控制添加。其中参数i基本可以理解为添加的第几层，因为其后调用时也基本就按照 0 1 2...的顺序。比如第i层，in=nm[i-1] out=nm[i] kernel_size=ks[i] stride=ss[i] padding=ps[i]，又因为in/out要比其他参数多一个，因此当i=0时，直接给in=1。然后判断需不需要batchNormalization和leakyRelu，如果需要的话就加这一层，如果不需要的话就直接加relu函数。

关于.format函数，其实是为了命名的，就是把format里面的参数放到前面的字符串中的括号里。直接贴菜鸟教程上的例子吧。
```python
>>>"{} {}".format("hello", "world")    # 不设置指定位置，按默认顺序
'hello world'
 
>>> "{0} {1}".format("hello", "world")  # 设置指定位置
'hello world'
 
>>> "{1} {0} {1}".format("hello", "world")  # 设置指定位置
'world hello world'
```

在后面添加模块时，基本上都有用到batchNormalization。具体这个是干啥的之后再仔细说。

进入forward之后。
