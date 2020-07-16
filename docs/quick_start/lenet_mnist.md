在这篇文章中，我们将学习：

* 使用oneflow接口配置软硬件环境

* 使用oneflow的接口定义训练模型

* 实现oneflow的训练作业函数

* 保存/加载模型训练结果

* 实现oneflow的校验作业函数

本文通过使用LeNet模型，训练MNIST数据集向大家介绍使用OneFlow的各个核心环节。
在文末附有完整的示例代码。

在学习之前，也可以通过以下命令查看各脚本功能。

首先，同步本文档仓库并切换到对应路径：
```shell
git clone https://github.com/Oneflow-Inc/oneflow-documentation.git
cd oneflow-documentation/docs/code/quick_start/
```

* 模型训练
```shell
python lenet_train.py
```
以上命令将对MNIST数据集进行训练，并保存模型。

训练模型是以下`lenet_eval.py`与`lenet_test.py`的前提条件，你也可以直接下载使用我们已经训练好的模型，略过训练步骤：
```shell
#在仓库docs/code/quick_start/目录下
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/lenet_models_1.zip
unzip lenet_models_1.zip
```

* 模型校验
```shell
python lenet_eval.py
```
以上命令，使用MNIST测试集对刚刚生成的模型进行校验，并给出准确率。

* 图像识别

```shell
python lenet_test.py ./9.png
```
以上命令将使用之前训练的模型对我们准备好的“9.png”图片中的内容进行预测。
你也可以下载我们[提取好的mnist图片](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/mnist_raw_images.zip)，自行验证自己训练模型的预测效果。

## MNIST数据集介绍

MNIST是一个手写数字的数据库。包括了训练集与测试集；训练集包含了60000张图片以及图片对应的标签，测试集包含了60000张图片以及图片测试的标签。Yann LeCun等已经将图片进行了大小归一化及居中处理，并且打包为二进制文件供下载。http://yann.lecun.com/exdb/mnist/

## 配置训练的软硬件环境

使用oneflow.function_config()可以构造一个配置对象，使用该对象，可以对训练相关的诸多软硬件参数进行配置。
与训练直接相关参数，被打包放置在`function_config`的train成员中，其余的配置直接作为`function_config`的成员。
以下是我们训练的基本配置：

```python
def get_train_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  config.train.primary_lr(0.1)
  config.train.model_update_conf({"naive_conf": {}})
  return config
```

在以上代码中，我们：

* 将训练的默认类型设置为float

* 设置learning rate为0.1

* 训练过程中的模型更新策略为"naive_conf"

config对象，其使用场景，将在后文 **实现训练任务函数** 中介绍。

## 定义训练模型

在oneflow.nn及oneflow.layers提供了用于构建模型的算子。

```python
def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(data, 32, 5, padding='SAME', activation=flow.nn.relu, name='conv1',
                               kernel_initializer=initializer)
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME', name='pool1')
    conv2 = flow.layers.conv2d(pool1, 64, 5, padding='SAME', activation=flow.nn.relu, name='conv2',
                               kernel_initializer=initializer)
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME', name='pool2', )
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name='dense1')
    if train: hidden = flow.nn.dropout(hidden, rate=0.5, name="dropout")
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name='dense2')

```

以上代码中，我们搭建了一个LeNet网络模型。

## 实现训练任务函数

OneFlow中提供了`oneflow.global_function`装饰器，通过它，可以将一个Python函数转变为训练任务函数（job function）。

### function装饰器

`oneflow.global_function`装饰器接收一个`function_config`对象作为参数。
它可以将一个普通Python函数转变为OneFlow的训练任务函数，并将前文所述的`function_config`所做的配置应用到其中。

```python
@flow.global_function(get_train_config())
def train_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE, ), dtype=flow.int32)):
    #任务函数实现 ...
```

### 指定优化特征
我们可以通过`oneflow.losses.add_loss`接口指定待优化参数。这样，OneFlow在每次迭代训练任务的过程中，将以该参数的最优化作为目标。

```python
@flow.global_function(get_train_config())
def train_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE, ), dtype=flow.int32)):
  with flow.fixed_placement("gpu", "0:0"):
    logits = lenet(images, train=True)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
  flow.losses.add_loss(loss)
  return loss
```

以上，我们通过`flow.nn.sparse_softmax_cross_entropy_with_logits`求得loss，并且将优化loss作为目标参数。

**注意** ：训练函数通过add_loss指定优化参数，与返回值并无关系。
以上示例中返回值是loss，但并不是必须，实际上训练函数的返回值主要用于训练过程中与外界交互。在下文中的 **调用任务函数并交互** 会详细介绍。

## 调用任务函数并交互

调用任务函数就可以开始训练。因为任务函数是经过OneFlow装饰器处理过的，因此返回的是OneFlow封装的对象，而 **不是** 之前定义任务函数时的返回值。
这涉及到了 **定义时的** 任务函数与 **调用时的** 任务函数如何交互问题。
方法很简单，**调用时** 返回的对象中，包括了`get`以及`async_get`方法，它们分别对应同步和异步。通过它们，我们可以获取 **定义时** 函数的返回值。

### 同步方式获取训练任务返回值

采用get方法，可以同步获取返回值数据。

```python
  for epoch in range(50):
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
      loss = train_job(images, labels).get().mean()
      if i % 20 == 0: print(loss)
```

以上代码中，使用`get`方法获取loss向量，然后计算平均值并打印。

### 异步方式获取训练任务返回值

`async_get`用于异步获取 **定义时** 训练任务函数的返回结果。
它需要我们准备一个回调函数，当OneFlow迭代完成训练任务函数时，会调用我们的回调函数，并且将训练任务函数的返回值作为参数传递给我们的回调函数。
代码示意：

```python
cb_handle_result(result):
    #...

job_func(images, labels).async_get(cb_handle_result)
```

具体的例子我们会在后文 **模型的校验** 中展示。

## 模型的初始化、保存与加载

### 模型的初始化与保存

`oneflow.train.CheckPoint`类构造的对象，可以用于模型的初始化、保存与加载。
在训练过程中，我们可以通过`init`方法初始化模型，通过`save`方法保存模型。如下例：

```python
if __name__ == '__main__':
  check_point = flow.train.CheckPoint()
  check_point.init()
  #加载数据及训练 ...  
  check_point.save('./lenet_models_1') 
```

保存成功后，我们将得到名为"lenet_models_1"的 **目录** ，该目录中包含了与模型参数对应的子目录及文件。

### 模型的加载

在校验或者预测过程中，我们可以通过`oneflow.train.CheckPoint.load`方法加载现有的模型参数。如下例：

```python
if __name__ == '__main__':
  check_point = flow.train.CheckPoint()
  check_point.load("./lenet_models_1")
  #校验过程 ...
```

load自动读取之前保存的模型，并加载。

## 模型的校验
校验任务函数与训练任务函数 **几乎没有区别** ，不同之处在于校验过程中的模型参数来自于已经保存好的模型，因此不需要初始化，也不需要在迭代过程中更新模型参数。

### 配置校验的软硬件环境

```python
def get_eval_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  return config
```

以上是校验过程的function_config配置，与训练过程相比，去掉了learning rate的选择，以及更新模型参数的设置。

### 校验任务函数的编写

```python
@flow.global_function(get_eval_config())
def eval_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE, ), dtype=flow.int32)):
  with flow.fixed_placement("gpu", "0:0"):
    logits = lenet(images, train=True)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

  return {"labels":labels, "logits":logits}
```

以上是校验训练任务函数的编写，返回了一个字典。我们将调用训练任务函数，并展示如何使用异步方式获取返回值。

### 迭代校验

准备回调函数：
```python
g_total = 0
g_correct = 0

def acc(eval_result):
  global g_total
  global g_correct

  labels = eval_result["labels"]
  logits = eval_result["logits"]

  predictions = np.argmax(logits.ndarray(), 1)
  right_count = np.sum(predictions == labels)
  g_total += labels.shape[0]
  g_correct += right_count
```

以上的回调函数`acc`，将被OneFlow框架调用，所得到的参数(`eval_result`)，就是训练任务函数的返回值。
我们在该函数中统计样本的总数目，以及校验正确的总数目。

调用校验任务函数：

```python
if __name__ == '__main__':
  check_point = flow.train.CheckPoint()
  check_point.load("./lenet_models_1")
  (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE, BATCH_SIZE)
  for epoch in range(1):
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
      eval_job(images, labels).async_get(acc)

  print("accuracy: {0:.1f}%".format(g_correct*100 / g_total))
```

以上，循环调用校验函数，并且最终输出对于测试集的判断准确率。

## 预测图片

将以上校验代码修改，使得校验数据来自于原始的图片而不是现成的数据集，我们就可以使用模型进行图片预测。

```python
def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = (im -128.0)/ 255.0
    im.reshape((-1, 1, 1, im.shape[1], im.shape[2]))
    return im

if __name__ == '__main__':
    check_point = flow.train.CheckPoint()
    check_point.load("./lenet_models_1")

    image = load_image(sys.argv[1])
    logits = eval_job(image, np.zeros((1,)).astype(np.int32)).get()
    
    prediction = np.argmax(logits.ndarray(), 1)
    print("predict:{}".format(prediction))
```

## 完整代码

### 训练模型

代码：[lenet_train.py](https://github.com/Oneflow-Inc/oneflow-documentation/blob/master/docs/code/quick_start/lenet_train.py)

```python
#lenet_train.py
import numpy as np
import oneflow as flow
from mnist_util import load_data
from PIL import Image
BATCH_SIZE = 100


def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(data, 32, 5, padding='SAME', activation=flow.nn.relu, name='conv1',
                               kernel_initializer=initializer)
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME', name='pool1')
    conv2 = flow.layers.conv2d(pool1, 64, 5, padding='SAME', activation=flow.nn.relu, name='conv2',
                               kernel_initializer=initializer)
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME', name='pool2', )
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name='dense1')
    if train: hidden = flow.nn.dropout(hidden, rate=0.5, name="dropout")
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name='dense2')


def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    config.train.primary_lr(0.1)
    config.train.model_update_conf({"naive_conf": {}})
    return config


@flow.global_function(get_train_config())
def train_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE,), dtype=flow.int32)):
    with flow.fixed_placement("gpu", "0:0"):
        logits = lenet(images, train=False)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
    flow.losses.add_loss(loss)
    return loss


def get_eval_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config


@flow.global_function(get_eval_config())
def eval_job(images=flow.FixedTensorDef((1, 1, 28, 28), dtype=flow.float),
             labels=flow.FixedTensorDef((1,), dtype=flow.int32)):
    with flow.fixed_placement("gpu", "0:0"):
        logits = lenet(images, train=False)
    return logits

if __name__ == '__main__':
    flow.config.gpu_device_num(1)
    check_point = flow.train.CheckPoint()
    check_point.init()

    (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE)

    for epoch in range(50):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            loss = train_job(images, labels).get().mean()
            if i % 20 == 0: print(loss)
            if loss < 0.01:
                break
    check_point.save('./lenet_models_1')  # need remove the existed folder
    print("model saved")
```

### 校验模型

代码：[lenet_eval.py](https://github.com/Oneflow-Inc/oneflow-documentation/blob/master/docs/code/quick_start/lenet_eval.py)

预训练模型：[lenet_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/lenet_models_1.zip)

```python
#lenet_eval.py
import numpy as np
import oneflow as flow
from mnist_util import load_data

BATCH_SIZE = 100


def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(data, 32, 5, padding='SAME', activation=flow.nn.relu, name='conv1',
                               kernel_initializer=initializer)
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME', name='pool1')
    conv2 = flow.layers.conv2d(pool1, 64, 5, padding='SAME', activation=flow.nn.relu, name='conv2',
                               kernel_initializer=initializer)
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME', name='pool2', )
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name='dense1')
    if train: hidden = flow.nn.dropout(hidden, rate=0.5)
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name='dense2')


def get_eval_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config


@flow.global_function(get_eval_config())
def eval_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
             labels=flow.FixedTensorDef((BATCH_SIZE,), dtype=flow.int32)):
    with flow.fixed_placement("gpu", "0:0"):
        logits = lenet(images, train=True)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

    return {"labels": labels, "logits": logits}


g_total = 0
g_correct = 0


def acc(eval_result):
    global g_total
    global g_correct

    labels = eval_result["labels"]
    logits = eval_result["logits"]

    predictions = np.argmax(logits.ndarray(), 1)
    right_count = np.sum(predictions == labels)
    g_total += labels.shape[0]
    g_correct += right_count


if __name__ == '__main__':

    check_point = flow.train.CheckPoint()
    check_point.load("./lenet_models_1")
    (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE, BATCH_SIZE)

    for epoch in range(1):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            eval_job(images, labels).async_get(acc)

    print("accuracy: {0:.1f}%".format(g_correct * 100 / g_total))
```

### 数字预测

代码：[lenet_test.py](https://github.com/Oneflow-Inc/oneflow-documentation/blob/master/docs/code/quick_start/lenet_test.py)

预训练模型：[lenet_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/lenet_models_1.zip)

MNIST数据集图片[mnist_raw_images.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/mnist_raw_images.zip)

```python
import numpy as np
import oneflow as flow
from PIL import Image

BATCH_SIZE = 1


def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(data, 32, 5, padding='SAME', activation=flow.nn.relu, name='conv1',
                               kernel_initializer=initializer)
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME', name='pool1')
    conv2 = flow.layers.conv2d(pool1, 64, 5, padding='SAME', activation=flow.nn.relu, name='conv2',
                               kernel_initializer=initializer)
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME', name='pool2', )
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name='dense1')
    if train: hidden = flow.nn.dropout(hidden, rate=0.5)
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name='dense2')


def get_eval_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config


@flow.global_function(get_eval_config())
def eval_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
             labels=flow.FixedTensorDef((BATCH_SIZE,), dtype=flow.int32)):
    with flow.fixed_placement("gpu", "0:0"):
        logits = lenet(images, train=False)
    return logits


def load_image(file):
    im = Image.open(file).convert('L')
    im = im.resize((28, 28), Image.ANTIALIAS)
    im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
    im = (im - 128.0) / 255.0
    im.reshape((-1, 1, 1, im.shape[1], im.shape[2]))
    return im


if __name__ == '__main__':

    check_point = flow.train.CheckPoint()
    check_point.load("./lenet_models_1")

    image = load_image("./9.png")
    logits = eval_job(image, np.zeros((1,)).astype(np.int32)).get()

    prediction = np.argmax(logits.ndarray(), 1)
    print("predict:{}".format(prediction[0]))
```
