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

In addition to mirroed strategy, OneFlow also provides consistent strategy. **Consistent strategy is the main characteristic of OneFlow, have lots of advantages compared with mirrored.(链接可能出错）**

OneFlow will use mirrored strategy as default. To use consistent strategy, we need use  `default_distribute_strategy` of `flow.function_config()` to config:
```python
  config = flow.function_config()
  config.default_distribute_strategy(flow.scope.consistent_view())
```

The reason of why consistent strategy is the main character of OneFlow is because in OneFlow design use `consistent_strategy`. Then from user's point of view, the op and blob can get consistently in logic level. We use matrixes multiplication as example in the beginning of article, we only need focus on [matrix multiplication](#mat_mul_op) itself on mathematics level. But in project, the issue of how to config and use model parallel or data parallel can be easily done by using OneFlow. OneFlow will handle **The data division of data parallel**, **model division of model parallel** and **serial logic** issue very fast and efficient.

 **In consistent strategy in OneFlow, we are free to choose either model parallel or data parallel or mix of them.**

### Example
In following script, we use consistent strategy and use two GPU to training. The default parallels method is **data parallel** in consistent strategy.The issue of how to set **model parallel** and **mix parallel** in consistent strategy will not be discussed in this article. We have special introduction of that in [parallels characters of OneFlow](model_mixed_parallel.md).

Name: [consistent_strategy.py](../code/extended_topics/consistent_strategy.py)

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

### Script explanation
In above script:

* Use  `flow.config.gpu_device_num` to define the GPU number:
```python
flow.config.gpu_device_num(2)
```

* Use  `oft.Numpy.Placeholder` to define the place holder in consistent strategy. Because the blob of `Numpy.Placeholder` is represent the op and place holder in logic. Thus. the BATCH_SIZE is the totally number of samples and no need for manually dividing or combining.
```python
def train_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
            labels:oft.Numpy.Placeholder((BATCH_SIZE, ), dtype=flow.int32))
```

* The result of training use `numpy` as a consistent results. OneFlow done the distributed training and merging process.In consistent model, multi card is basically no difference with single card. User will not find that when using.
```python
      loss = train_job(images, labels).get().ndarray()
      if i % 20 == 0: 
        print(loss.mean())
```

## More extending
With the development of machine learning theory and practice, and now there are already many network unable to training by single card. There have been more and more using of data can not be trained on the parallel model.

Use  `consistent` in OneFlow, by free to choice and combination of parallel. Can give a good solutions to the above issue. We will introduce in [parallel characteristic of OneFlow](model_mixed_parallel.md).


