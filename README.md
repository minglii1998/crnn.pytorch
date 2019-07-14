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

