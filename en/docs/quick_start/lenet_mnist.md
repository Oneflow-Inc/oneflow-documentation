In this article, we will learn:

* 使用 oneflow 接口配置软硬件环境

* 使用 oneflow 的接口定义训练模型

* 实现 oneflow 的训练作业函数

* Save/load the result of training model.

* 实现 oneflow 的校验作业函数

本文通过使用 LeNet 模型，训练 MNIST 数据集向大家介绍使用 OneFlow 的各个核心环节。 The full example code is attached in the end.

Before learning, you can check the function of each script by running the following command.

First of all, clone the documentation repository and switch to the corresponding path:
```shell
git clone https://github.com/Oneflow-Inc/oneflow-documentation.git
cd oneflow-documentation/docs/code/quick_start/
```

* Training model
```shell
python lenet_train.py
```
以上命令将对 MNIST 数据集进行训练，并保存模型。

训练模型是以下 `lenet_eval.py` 与 `lenet_test.py` 的前提条件，你也可以直接下载使用我们已经训练好的模型，略过训练步骤：
```shell
#Repository location: docs/code/quick_start/ 
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/lenet_models_1.zip
unzip lenet_models_1.zip
```

* Model evaluation
```shell
python lenet_eval.py
```
以上命令，使用 MNIST 测试集对刚刚生成的模型进行校验，并给出准确率。

* Image recognition

```shell
python lenet_test.py ./9.png
```
以上命令将使用之前训练的模型对我们准备好的 “9.png” 图片中的内容进行预测。 Or we can download our [ prepared image](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/mnist_raw_images.zip)to verify their training model prediction result.

## MNIST dataset introdaction

MNIST 是一个手写数字的数据库。Training set include 60000 pictures and the corresponding label.Yann LeCun 等已经将图片进行了大小归一化及居中处理，并且打包为二进制文件供下载。http://yann.lecun.com/exdb/mnist/

## Configuration of hardware and software training environment

使用 `oneflow.function_config()` 可以构造一个配置对象，使用该对象，可以对训练相关的诸多软硬件参数进行配置。 与训练直接相关参数，被打包放置在 `function_config` 的 train 成员中，其余的配置直接作为 `function_config` 的成员。 Following is our basic configuration of training:

```python
def get_train_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  config.train.primary_lr(0.1)
  config.train.model_update_conf({"naive_conf": {}})
  return config
```

In the code above:

* 将训练的默认类型设置为 float

* 设置 learning rate 为0.1

* 训练过程中的模型更新策略为 "naive_conf"

config 对象，其使用场景，将在后文 **实现训练任务函数** 中介绍。

## Define training model

在 `oneflow.nn` 及 `oneflow.layers` 提供了部分用于构建模型的算子。

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

以上代码中，我们搭建了一个 LeNet 网络模型。

## Implement training function

OneFlow 中提供了 `oneflow.global_function` 装饰器，通过它，可以将一个 Python 函数转变为训练任务函数（job function）。

### Function decorator

`oneflow.global_function` 装饰器接收一个 `function_config` 对象作为参数。 它可以将一个普通 Python 函数转变为 OneFlow 的训练任务函数，并将前文所述的 `function_config` 所做的配置应用到其中。

```python
@flow.global_function(get_train_config())
def train_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:oft.Numpy.Placeholder((BATCH_SIZE, ), dtype=flow.int32)):
    #任务函数实现 ...
```

### Specify the optimization feature
我们可以通过 `oneflow.losses.add_loss` 接口指定待优化参数。We can using `oneflow.losses.add_loss`'s port to specify the parameters which need to optimization.In this way, OneFlow will trade optimise the parameter as target when each iteration training mission.

```python
@flow.global_function(get_train_config())
def train_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:oft.Numpy.Placeholder((BATCH_SIZE, ), dtype=flow.int32)):
  with flow.scope.placement("gpu", "0:0"):
    logits = lenet(images, train=True)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
  flow.losses.add_loss(loss)
  return loss
```

以上，我们通过 `flow.nn.sparse_softmax_cross_entropy_with_logits` 求得 loss ，并且将优化 loss 作为目标参数。

**注意** ：训练函数 add_loss 所指定的优化参数，与返回值没有关系。 以上示例中返回值是 loss ，但并不是必须，实际上训练函数的返回值主要用于训练过程中与外界交互。We will introduce that in more details in **Call the job function and interaction**below.

## Call the jon function and interaction

We can start training when called the job function.因为任务函数是经过 OneFlow 装饰器处理过的，因此返回的是 OneFlow 封装的对象，而 **不是** 之前定义任务函数时的返回值。 This involved a issue which how to interact **previous defined** job function and **previous called** job function. 方法很简单，**调用时** 返回的对象中，包括了 `get` 以及 `async_get` 方法，它们分别对应同步和异步。Through them, we can obtained return value of function when **defined them**.

### Synchronously method obtained the return value when training task

采用 get 方法，可以同步获取返回值数据。

```python
  for epoch in range(50):
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
      loss = train_job(images, labels).get().mean()
      if i % 20 == 0: print(loss)
```

以上代码中，使用 `get` 方法获取 loss 向量，然后计算平均值并打印。

### Asynchronous method obtained the return value when training task

`async_get` 用于异步获取 **定义时** 训练任务函数的返回结果。 它需要我们准备一个回调函数，当 OneFlow 迭代完成训练任务函数时，会调用我们的回调函数，并且将训练任务函数的返回值作为参数传递给我们的回调函数。 Sample code:

```python
cb_handle_result(result):
    #... job_func(images, labels).async_get(cb_handle_result)
```

More details example will be demonstrated in **Model evaluation**.

## Initialization, saving and loading models

### Initialization and saving model

`oneflow.train.CheckPoint` 类构造的对象，可以用于模型的初始化、保存与加载。 在训练过程中，我们可以通过 `init` 方法初始化模型，通过 `save` 方法保存模型。For example:

```python
if __name__ == '__main__':
  check_point = flow.train.CheckPoint()
  check_point.init()
  #load data and training ...  
  check_point.save('./lenet_models_1') 
```

保存成功后，我们将得到名为 "lenet_models_1" 的 **目录** ，该目录中包含了与模型参数对应的子目录及文件。

### Loading models

在校验或者预测过程中，我们可以通过 `oneflow.train.CheckPoint.load` 方法加载现有的模型参数。For example:

```python
if __name__ == '__main__':
  check_point = flow.train.CheckPoint()
  check_point.load("./lenet_models_1")
  #evaluation process  ...
```

load 自动读取之前保存的模型，并加载。

## Evaluation of models
Evaluation job function **basically is same as** train job function. The difference is in evaluation process, the model we use is already saved. Thus, do not require initialize and update model during Iteration.

### Configure the hardware and software environment of evaluation

```python
def get_eval_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  return config
```

以上是校验过程的 function_config 配置，与训练过程相比，去掉了 learning rate 的选择，以及更新模型参数的设置。

### Coding of evaluation job function

```python
@flow.global_function(get_eval_config())
def eval_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:oft.Numpy.Placeholder((BATCH_SIZE, ), dtype=flow.int32)):
  with flow.scope.placement("gpu", "0:0"):
    logits = lenet(images, train=True)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

  return {"labels":labels, "logits":logits}
```

Above is the coding of evolution job function and return object is a dictionary.Above is the coding of evolution job function and return object is a dictionary.We will call train job function and demonstrated how to use asynchronous method to obtain return value.

### Iteration evaluation

Prepare callbcak function:
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

以上的回调函数 `acc` ，将被 OneFlow 框架调用，所得到的参数(`eval_result`)，就是训练任务函数的返回值。 We record the total number of sample and total correct results number of sample in this function.

Called evacuation job function:

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

So far, Cycle call the evaluation function and output the accuracy of result of testing set.

## Image prediction

Modify the above evaluation code, change the evaluate date to raw images rather than the existing dataset. Then we can use model to predict the content in the image.

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

## Complete code

### Training model

Name: [lenet_train.py](https://github.com/Oneflow-Inc/oneflow-documentation/blob/master/docs/code/quick_start/lenet_train.py)

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
def train_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:oft.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)):
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=False)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
    flow.losses.add_loss(loss)
    return loss


def get_eval_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config


@flow.global_function(get_eval_config())
def eval_job(images:oft.Numpy.Placeholder((1, 1, 28, 28), dtype=flow.float),
             labels:oft.Numpy.Placeholder((1,), dtype=flow.int32)):
    with flow.scope.placement("gpu", "0:0"):
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

### Evaluate model

Name: [lenet_eval.py](https://github.com/Oneflow-Inc/oneflow-documentation/blob/master/docs/code/quick_start/lenet_eval.py)

Saved model: [lenet_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/lenet_models_1.zip)

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
def eval_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
             labels:oft.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)):
    with flow.scope.placement("gpu", "0:0"):
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

### Number prediction

Name: [lenet_test.py](https://github.com/Oneflow-Inc/oneflow-documentation/blob/master/docs/code/quick_start/lenet_test.py)

Saved model: [lenet_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/lenet_models_1.zip)

MNIST 数据集图片[mnist_raw_images.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/mnist_raw_images.zip)

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
def eval_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
             labels:oft.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)):
    with flow.scope.placement("gpu", "0:0"):
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
