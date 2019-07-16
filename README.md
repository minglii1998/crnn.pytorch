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


- [1. 训练](#1-训练)
- [2. 代码解释](#2-代码解释)
    - [2.1 models/crnn.py](#21-modelscrnnpy)
    - [2.2 train.py](#22-trainpy)
        - [2.2.1 val()](#221-val)
        - [2.2.2 trainBatch()](#222-trainbatch)
        - [2.2.3 for epoch](#223-for-epoch)
    - [2.3 dataset.py](#23-datasetpy)
        - [2.3.1 class lmdbDataset(Dataset)](#231-class-lmdbdatasetdataset)
        - [2.3.2 class resizeNormalize()](#232-class-resizenormalize)
        - [2.3.3 class alignCollate()](#233-class-aligncollate)
    - [2.4 utils.py](#24-utilspy)
        - [2.4.1 class averager()](#241-class-averager)
        - [2.4.2 encode & decode](#242-encode--decode)
        - [2.4.3 loadData(v, data)](#243-loaddatav-data)
- [3 测试](#3-测试)
    - [3.1 遍历单个文件夹](#31-遍历单个文件夹)
    - [3.2 遍历文件夹及其子文件夹](#32-遍历文件夹及其子文件夹)
    - [3.3 lmdb数据库](#33-lmdb数据库)
- [4 总结](#4-总结)



## 1. 训练

训练要用`create_dataset.py`来制作lmdb格式的数据库。但是由于不知道什么的原因，用自己下载的svt数据库然后转成lmdb时遇到了问题，这个问题暂时还未解决，但是有MJ的lmdb数据库，这个问题暂时不考虑。

在试图训练的过程中，会遇到一些bug，不知道是出于什么样的原因，代码原作者并没有去修改这些bug。我已在`zz bug&handling--Ming.txt`中记录下了这些bug，因此遇到问题直接对照该文档即可，就不再把它搬到readme里了。

## 2. 代码解释

###### 自己只是初学者，只是在自己所理解的范围内稍加解释。

### 2.1 models/crnn.py

在自己的另一个仓库里，有一个[CRNN Learning Note](https://github.com/minglii1998/TextDetection/blob/master/CRNN%20Learning%20Note.md)，可以对照着它来看关于crnn的模型。

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

进入forward之后，就直接conv = self.cnn(input)，然后转换一下维度之后，再output = self.rnn(conv)就ok了。这个写法写的要比教程中的更好理解一点。相当于官网教程是把卷积层、连接层这种红层和池化层，激活层分开了，前面定义卷积、链接，后面在forward中定义池化和激活。还是有点意思的。

然后在这里class BidirectionalLSTM，也同理得很简单..双向就根本不用额外多写什么，就直接nn.LSTM(nIn, nHidden, bidirectional=True)把bidirectiona这个参数设置为True就好了。

### 2.2 train.py

这里parser的参数就不多介绍了，基本上都是可以望文生义的。

```python
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
test_dataset = dataset.lmdbDataset(
    root=opt.valRoot, transform=dataset.resizeNormalize((100, 32)))
```

* 这里是把训练集和测试集都放到loader里面去，loader可以理解为一个迭代器
* `collate_fn` (callable, optional): merges a list of samples to form a mini-batch.

```python
nclass = len(opt.alphabet) + 1
nc = 1

converter = utils.strLabelConverter(opt.alphabet)
criterion = CTCLoss()
```
* 这里的alphabet就是字母表，需要机器能够辨别的所有字符，准确的说应该是产生的字符，默认是数字和小写字母
* converter是吧alphabet这个大的字符串变成列表

```python
# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
crnn.apply(weights_init)
```
* 这里是在对网络的参数进行初始化，一般单层网络可直接用torch.nn.innit，多层网络则利用torch.nn.Module.apply
* 具体步骤：
   1. 定义weight_init函数，并在weight_init中通过判断模块的类型来进行不同的参数初始化定义类型。
   2. model=Net(…) 创建网络结构。 
   3. model.apply(weight_init),将weight_init初始化方式应用到submodels上
* 关于Variable
   Variable封装了tensor，并记录对tensor的操作记录用来构建计算图。主要含有3个属性：
   data：保存Variable中包含的tensor
   grad：保存data对应的梯度，grad也是Variable而不是tensor，它与data形状一致
   grad_fn：指向一个Functionn，记录tensor的操作历史，即它是什么操作的输出，用来构建计算图。

#### 2.2.1 val()

这个函数时用来测试结果如何的，由于每次开始前都会有输出，所以还比较好分辨。

```python
    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)
```
* 这里把requires_grad设置成了False，就不让它继续算梯度了，也就是现在的测试部分并不影响训练
* model.eval()，让model变成测试模式，对dropout和batch normalization的操作在训练和测试的时候是不一样的

```python
    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.squeeze()
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1
```
* 一大坨看了让人自闭，先放着吧
* 

#### 2.2.2 trainBatch()

```python
def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    optimizer.step()
    return cost
```
* 上面那坨就是闲的蛋疼...从iterate里获得data后放到cpu_data中，然后把text编码后，放到了t,l这两个中间变量，然后又把它们复制到了text和length，感觉有点冗余这样子，除非我是没有理解loadData是干啥的。
* 下面这坨就是训练的部分了，根据官网那个简单的例子，整个应该是：
   1. zero the parameter gradients
   2. outputs = net(inputs)
   3. forward + backward + optimize
   然后这里的顺序是213，不过应该都差不多吧。

#### 2.2.3 for epoch

```python
for epoch in range(opt.nepoch):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()
```
* 这里确定了循环的计数器，最外面的epoch是表示要训练几轮，里面的while是用来计数每次epoch里有多少图片
* 除此之外就是得到参数，设置为可以计算梯度，以及用train()进入训练模式
* 主要的训练还是在上面的trainBatch()中

### 2.3 dataset.py

#### 2.3.1 class lmdbDataset(Dataset)

这边提到了lmdb，我之前也有大概的看一下lmdb的基本操作，具体用法见[Python lmdb使用](https://blog.csdn.net/u010472607/article/details/76855509)。

```python
# 归纳一下lmdb的大概用法

# 打开环境
# 第一个是路径，第二个是最大储存容量，单位是kb，以下定义1TB容量
env = lmdb.open("./train"，map_size=1099511627776)

# 建立事务
# 不知道这个txn代表什么，但是已经约定俗成用txn了
txn = env.begin(write=True)

txn.put(key, value) # 插入和修改
txn.delete(key)# 进行删除
txn.get(key)# 进行查询
txn.cursor()# 进行遍历
txn.commit()# 提交更改

# 最后关闭环境
env.close()

```

可以看到，在获取lmdb里的数据时，键值都需要encode，因为lmdb存储的数据用的是二进制，因此不encode的话找不到相应的键值。

__getitem__(self, index)这个函数的作用是从lmdb里面取出一个数据，返回一个(img, label)。

```python
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = txn.get(label_key.encode()).decode()

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)
```
* 这里的imgbuf存储的是图片的二进制形式，然后用write写入buf中，再用seek(0)获得buf这个file的起始处，然后打开
* `six.BytesIO()`:Buffered I/O implementation using an in-memory bytes buffer.
* `write()`:Write bytes to file.
* `seek()`:Change stream position. 0:Start of stream (the default). Returns the new absolute position.
* `label = txn.get(label_key.encode()).decode()`这里也比较有趣，原git没有用decode，而是用str()，这个就造成了很烦的问题，具体可见txt文档。这里直接得到的label也应该是二进制的，因此需要decode一下，如果用str，就会直接变成刑辱b'xxx'这样的字符串，就无法得到真的label。
* 就是这个函数卡了我很久，第一次是用svt训练集，一直没办法读图片，第二次是会出现keyerror，之后再用svt数据集，看看不能读图片的问题能不能解决吧
* 不过这里也有个问题，transform函数在哪里定义啊？咋没找到...

#### 2.3.2 class resizeNormalize()

这个类的目的是为了把图片改成标准大小的。

```python
    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img
```
* `resize()`函数没找到更详细的定义，但是我自己试了下，resize((width,height))，不是切割，而是直接把图片拉变形，但是没懂interpolation是干啥的
* `img.sub_(0.5).div_(0.5)`这个没懂什么意思，我将Img用该函数处理后得到Img2，发现两者形状一样，并且值也一样，不知道是干啥的

#### 2.3.3 class alignCollate()

大概如果想弄懂这个是干啥的，还是得搞清楚data loader里的collate_fn是干啥的。

```python
    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels
```
* `zip()`：这个可以参考菜鸟教程上的例子，zip(* )相当于把原来的迭代器解压
* 在不需要等比例的情况下，直接把所有的img都用上面的resizeNormalize方法给弄成标准的样子
* `torch.cat()`是用来拼接张量的，第二个参数决定是按照行还是列来拼接
* 如果需要按比例的话，那就先找到w/h中最大的为max_ratio，（最扁的），总之就是弄成最扁的就对了..

### 2.4 utils.py

#### 2.4.1 class averager()

注释说是用来计算tensor和Variable的平均。其中add为加，val为计算均值。

`torch.numel()` ：返回一个tensor变量内所有元素个数，可以理解为矩阵内元素的个数
`torch.sum() `：返回输入向量input中所有元素的和。

#### 2.4.2 encode & decode

首先要解释一下，这里的encode和decode其实和python里原来的encode和decode无关，原来的是把字符串改成二进制型的，在这里并不是。

```python
    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))
```
* 当输入为字符串时，将字符串中的每个字符都找到与字母表中对应的，然后放入列表，也就是说是把字符串转换成了字符的列表。若输入的是list of str，就会把所有的字符串连在一起，然后再变成列表，其中返回的length为字符的个数而不是字符串的个数。

#### 2.4.3 loadData(v, data)

把data的数据烤到v中。

btw，我好像没看到oneHot这个函数有啥用诶？不管了。

## 3 测试

测试数据的话应该主要就是修改demo.py，目前demo可以做到的是对一个输入图片进行输出。那我现在测试就只要把所有的测试图片全部输进去，然后运行，看运行出来的结果和标签一样不一样，然后计算个数最后算准确率。

理论上，现在只要得到一个全是图片途径的列表，再得到一个label的列表，然后分别比较就好了。那么问题在于，如何得到图片的路径及标签呢？

### 3.1 遍历单个文件夹

对于某些文件夹，里面直接包含了图片，然后图片名称为label，可以直接用之前的代码把他们提取出来放在list里.（尴尬我把之前的代码删了..）

```python
test_path = '/home/liming/data/MJSynth/ramdisk/max/90kDICT32px/1/1/'

test_list = glob.glob(os.path.join(test_path, '*.jpg'))
label_list = []

for item in test_list:
    label_list.append(item.split('_')[1].lower())
```

这个可以把文件夹下所有的jpg格式的图片的路径存到list里，然后用split函数分割就可以得到label。

注意split()的用法，直接str.split(char,int)，第一个参数为分隔符，第二个参数为分为几个参数列表（这里没用到第二个参数）。

### 3.2 遍历文件夹及其子文件夹

比如还是MJ，每个文件夹里面是很好分，但是不知道为什么把所有的图片文件都分开放了。比如文件夹1下面又有10个文件夹，然后每个文件夹下才是照片。这种的话就需要迭代一下了，需要把某一个文件夹下的所有子文件夹里的照片全得到。

其实用上面的方法好像也是可以的，无非多写两层..我先查查有没有其他api吧。

```python
g = os.walk(r"/home/liming/data/MJSynth/ramdisk/max/90kDICT32px/1")  

for path,dir_list,file_list in g:  
    for file_name in file_list:  
        print(os.path.join(path, file_name) )

```
用这个可以获得某个文件夹下的所有子文件。然后那个walk()里的参数用单引号括起来不行，会报错`SyntaxError: EOL while scanning string literal`，所以就改成这种r""的模式了。

```python
test_path = r"/home/liming/data/MJSynth/ramdisk/max/90kDICT32px/1"
g = os.walk(test_path)
test_list = []
label_list = []

for path,dir_list,file_list in g:  
    new_list = glob.glob(os.path.join(path, '*.jpg'))
    test_list = test_list + new_list

for item in test_list:
    label_list.append(item.split('_')[1].lower())
```

上面这样就可以得到某一个文件夹下所有的图片了，应该说第一种是第二种的特例吧。

然后这边要注意的是，新加了一个new_list，因为glob.glob(os.path.join(path, '*.jpg'))直接创造了一个新的list，如果还像原来那样的话，就会出现test_list在每次遍历后都会被替换为新的，就只能得到最后子文件夹下的图片。代码为`test_file.py`。

### 3.3 lmdb数据库

用lmdb的主要是多了一步要从lmdb中提取出数据。然后由于提取的图片是没有path的，因此不可以像在文件夹里那样子获取label list和图片的path list然后直接用demo的api。

因为数据库中有数据的总数，所以，可以先获取总数，然后用for循环对每一个进行判断。这里要注意一个坑，range是从0开始的，而0是没有label和image数据的，因此range的范围应该是(1,nSamples+1)。具体看代码吧`test_lmdb.py`。

## 4 总结

这是次写readme是把这个代码详详细细地读了一遍。基本在读的时候是没什么问题的。如果需要写的话应该也可以大概按照找个抄抄然后缝缝补补修修改改能写出来吧？
不过之后是的确还得继续多看相关的代码。


