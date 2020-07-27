When doing distributed training, OneFlow provide two aspects for determining the relationship between data and models. There are `consistent` strategy and `mirrored` strategy.

In this article, we will introduce:

* The difference and applicable scenario of the data parallel and model parallel.

* The characteristics of using  `mirrored`  in distributed training.

* The characteristics of using  `consistent` in distributed training.

## Data parallel and model parallel.
In order to better understand  `consistent` and `mirrored` in OneFlow. We need to understand the difference between **data parallel **and **model parallel** in distributed training.

To further demonstrated the difference between data parallel and model parallel, we will introduce a simple operator(In OneFlow, the logical calculation will regard as operator): matrix multiplication

We assume that in training model have a matrix I as input. Multiply matrix I and W then get result O.

![I×W](imgs/i_mul_w.png)

As the description above, size of I is (N, C1), size of W is (C1, C2) and size of O is (N, C2).

Combined  machine learning logic. It can give definitions to the matrixes above:

* Matrix I as the input object, each line is a sample and each column is represents the characteristics of sample.

* Matrix W represents the parameters of model.

* Matrixe O is the result of prediction or label. If it is a prediction task. Then it is a process of figure out O by I and W and get the distribution result. If it is a training task. Then Then it is a process of figure out W by I and O.

When the line N in matrixe I is very large. It means we have large scale of sample. If when C2 in matrixe W is very large. It means we have very complex model. If the scale and complexity reached a point. The solo machine with solo GPU will not able to handle the training job. We might consider the distributed training. In distributed training system, we can choose ** data parallel **and **model parallel**.

<a id="mat_mul_op"></a>
In order to better understand data parallel and model parallel, we use the following figure as the demo of matrix multiplication operator:

![mat_mul_op](imgs/mul_op_illustrated.png)

The first matrixe in grey on left of equation is the input sample. Each line is a sample. The second matrixe in blue on left of equation is the model.

In this article, we will see the operators above switching to different way under data parallel and model parallel.


### Data parallel diagram

In **data parallel**, divide the sample data in small parts. **Data after dividing **will send to each training nodes and calculate with the **completely models**. Finally combined the information in each nodes. Like figure show below:

![mat_mul_op](imgs/mul_op_data_parr.png)

### Model parallel diagram

In **model parallel**, model will be divided. **Completely data** will send to each nodes and calculate with **model after dividing**. Finally combined the model in each nodes. Like figure show below:

![mat_mul_op](imgs/mul_op_model_parr.png)

Basically:

* In data parallel, each node use the same model to train, data will be cut.

* In model parallel, each node received same data, model will be cut.

We will introduce two parallel strategies in OneFlow (`mirrored` and `consistent`). Study how to choose different parallel methods in different strategies.

### Two type of place holder
In [use OneFlow build neural network](../basics_topics/build_nn_with_op_and_layer.md), we already introduce the concept of  `Placeholder`. It is data place holder.

In fact, for parallel, the  `Placeholder`  of OneFlow can divide to two types: Use `oneflow.typing.Numpy.Placeholder` and `oneflow.typing.ListNumpy.Placeholder` to constructing the place holder is corresponding to `Consistent`  and `Mirrored`.

We will see the detailed examples below.


## Using mirrored in OneFlow

Other framework like TensorFlow or Pytorch are support mirroed strategy. The strategy of OneFlow is similar to them.

In mirrored, the model in each nodes is same, thus we only can use **data parallel**.

In OneFlow, the default strategy is mirrored or you can use `default_distribute_strategy` of  `flow.function_config()` to define:

```python
    func_config = flow.function_config()
    func_config.default_distribute_strategy(flow.scope.mirrored_viewed())
```

In `mirrored_strategy`, only can use **data parallel**. When calling the function, we need divided data in average according to number of the GPU and put the data after dividing in to `list`. Every elements in `list` is the data to send to **each GPU**.

The return value of job function no longer use  `numpy` data. It actually use `numpy_list`  to get a  `list`. Every elements in  `list`is corresponding to the results of each GPU.

**Combined all **elements `list` mention above can make a complete BATCH.

### Example
In following script, we use default  `mirrored_strategy` strategy with two GPU to training.

Name: [mirrored_strategy.py](../code/extended_topics/mirrored_strategy.py)

The key part of the description in "script explanation" section.

```python
import numpy as np
import oneflow as flow
from mnist_util import load_data
import oneflow.typing as oft


BATCH_SIZE = 100
GPU_NUM = 2
BATCH_SIZE_PER_GPU  = int(BATCH_SIZE/GPU_NUM)

def get_train_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  config.train.primary_lr(0.1)
  config.train.model_update_conf({"naive_conf": {}})
  return config


@flow.global_function(get_train_config())
def train_job(images:oft.ListNumpy.Placeholder((BATCH_SIZE_PER_GPU, 1, 28, 28), dtype=flow.float),
              labels:oft.ListNumpy.Placeholder((BATCH_SIZE_PER_GPU, ), dtype=flow.int32)):
  initializer = flow.truncated_normal(0.1)
  reshape = flow.reshape(images, [images.shape[0], -1])
  hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer)
  logits = flow.layers.dense(hidden, 10, kernel_initializer=initializer)
  loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
  flow.losses.add_loss(loss)
  return loss


if __name__ == '__main__':
  flow.config.gpu_device_num(2) #设置GPU数目
  check_point = flow.train.CheckPoint()
  check_point.init()
  (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE)

  for i, (images, labels) in enumerate(zip(train_images, train_labels)):
    images1 = images[:BATCH_SIZE_PER_GPU]
    images2 = images[BATCH_SIZE_PER_GPU:]
    labels1 = labels[:BATCH_SIZE_PER_GPU]
    labels2 = labels[BATCH_SIZE_PER_GPU:]

    imgs_list = [images1, images2]
    labels_list = [labels1, labels2]

    loss = train_job(imgs_list, labels_list).get().ndarray_list()
    total_loss = np.array([*loss[0], *loss[1]])
    if i % 20 == 0: print(total_loss.mean())
```

### Script explanation
In the above script:

* Use  `flow.config.gpu_device_num` to set GPU amount as two.
```python
flow.config.gpu_device_num(2)
```

* `oneflow.typing.ListNumpy.Placeholder` defined the sample amount which is the amount after dividing. And the relationship between `BATCH_SIZE_PER_GPU`  and `BATCH_SIZE` is `BATCH_SIZE=BATCH_SIZE_PER_GPU×GPU_NUM`.
```python
def train_job(images:oft.ListNumpy.Placeholder((BATCH_SIZE_PER_GPU, 1, 28, 28),       dtype=flow.float),
        labels:oft.ListNumpy.Placeholder((BATCH_SIZE_PER_GPU, ), dtype=flow.int32))
```

* The data after dividing need to store in the  `list` and pass to training functions. The number of elements in `list` need be same as the **GPU number in training**. OneFlow will pass the data according to the order of the elements in `list ` to each GPU(the number i element in `list` is corresponding to number i GPU):
```python
    images1 = images[:BATCH_SIZE_PER_GPU]
    images2 = images[BATCH_SIZE_PER_GPU:]
    labels1 = labels[:BATCH_SIZE_PER_GPU]
    labels2 = labels[BATCH_SIZE_PER_GPU:]

    imgs_list = [images1, images2]
    labels_list = [labels1, labels2]

    loss = train_job(imgs_list, labels_list).get().ndarray_list()
```

* The results use  `numpy_list`  convert to numpy data which is also a list. The number of elements in this `list` need be same as **the number of GPU in training process**. Then we do the combination then print the  `total_loss`
```python
    loss = train_job(imgs_list, labels_list).get().ndarray_list()
    total_loss = np.array([*loss[0], *loss[1]])
    if i % 20 == 0: print(total_loss.mean())
```

## Use consistent strategies in OneFlow
We already know the mirrored strategy. In `mirrored_view`, sample will assign to many exactly same model to training distributed. The results of each nodes need be assembled to get the completed batch.

除了 mirroed 策略外，OneFlow 还提供了 consistent 策略。 **consistent 策略是 OneFlow 的一大特色，与 mirrored 策略相比有很大的优势。**

默认情况下 OneFlow 采取的是 mirrored 策略，使用 consistent 策略需要通过 `flow.function_config()` 的 `default_distribute_strategy` 接口显式设置：
```python
  config = flow.function_config()
  config.default_distribute_strategy(flow.scope.consistent_view())
```

之所以说 consistent 策略是 OneFlow 的一大特色，是因为在 OneFlow 的设计中，若采用 `consistent_strategy`，那么从用户的视角看，所使用的op、blob将获得 **逻辑上的统一**，同样以本文开头的矩阵乘法为例，我们只需要关注[矩阵乘法](#mat_mul_op)本身数学计算上的意义；而在工程上到底如何配置、采用模型并行还是数据并行等细节问题，可以使用OneFlow的接口轻松完成。OneFlow内部会高效可靠地解决 **数据并行中的数据切分** 、**模型并行中的模型切分** 、**串行逻辑** 等问题。

 **在OneFlow的consistent策略下，可以自由选择模型并行、数据并行或者两者共存的混合并行。**

### 代码示例
以下代码，我们采用 consistent 策略，使用2个 GPU 进行训练，consistent 策略下默认的并行方式仍然是 **数据并行**。关于如何在consistent 策略下设置 **模型并行** 及 **混合并行** 不在本文讨论范围，我们在[OneFlow的并行特色](model_mixed_parallel.md)中有专门的介绍与示例。

完整代码：[consistent_strategy.py](../code/extended_topics/consistent_strategy.py)

```python
import numpy as np
import oneflow as flow
from mnist_util import load_data
import oneflow.typing as oft


BATCH_SIZE = 100

def lenet(data, train=False):
  initializer = flow.truncated_normal(0.1)
  conv1 = flow.layers.conv2d(data, 32, 5, padding='SAME', activation=flow.nn.relu,
                             kernel_initializer=initializer, name="conv1")
  pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME', name="pool1")
  conv2 = flow.layers.conv2d(pool1, 64, 5, padding='SAME', activation=flow.nn.relu,
                             kernel_initializer=initializer, name="conv2")
  pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME', name="pool2")
  reshape = flow.reshape(pool2, [pool2.shape[0], -1])
  hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name="hidden")
  if train: hidden = flow.nn.dropout(hidden, rate=0.5)
  return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="outlayer")


def get_train_config():
  config = flow.function_config()
  config.default_distribute_strategy(flow.scope.consistent_view())
  config.default_data_type(flow.float)
  config.train.primary_lr(0.1)
  config.train.model_update_conf({"naive_conf": {}})
  return config


@flow.global_function(get_train_config())
def train_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:oft.Numpy.Placeholder((BATCH_SIZE, ), dtype=flow.int32)):
  logits = lenet(images, train=True)
  loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
  flow.losses.add_loss(loss)
  return loss


if __name__ == '__main__':
  flow.config.gpu_device_num(2)
  check_point = flow.train.CheckPoint()
  check_point.init()
  (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE)

  for epoch in range(50):
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
      loss = train_job(images, labels).get().numpy()

      if i % 20 == 0: 
        print(loss.mean())
```

### 代码解析
以上代码中：

* 使用 `flow.config.gpu_device_num` 设置GPU数目：
```python
flow.config.gpu_device_num(2)
```

* 使用 `oft.Numpy.Placeholder` 定义 consistent 策略下的占位符，因为`Numpy.Placeholder`产出的blob代表逻辑上的op及数据占位符，因此此处的BATCH_SIZE就是整个分布式训练的样本总和，不需要人为切分或者组合
```python
def train_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels:oft.Numpy.Placeholder((BATCH_SIZE, ), dtype=flow.int32))
```

* 训练结果通过 `numpy` 转化为 numpy 数据，直接是一个统一的结果，OneFlow 完成分布式过程中切分与合并的工作。在 consistent 模式下，多卡的分布式训练与单卡的训练，代码差别极少，上手体验几乎一样
```python
      loss = train_job(images, labels).get().ndarray()
      if i % 20 == 0: 
        print(loss.mean())
```

## 更多扩展
随着机器学习理论与实践发展，现在已经出现了很多单机无法训练的网络；也出现了越来越多仅采用数据并行无法训练的模型。

采用OneFlow的 `consistent` 策略，通过自由选择及组合并行方式，可以很好地解决以上问题，我们在[OneFlow的并行特色](model_mixed_parallel.md)进行了专门的介绍。


