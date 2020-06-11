
我们在这篇文章中将学会：

* 使用OneFlow内置函数加载数据

* 进行图片分类训练

* 实时打印训练相关结果

## 代码
运行以下代码，通过mlp模型，训练mnist数据集。
```python
import numpy as np
import oneflow as flow
from mnist_util import load_data


BATCH_SIZE = 100

def get_train_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  config.train.primary_lr(0.1)
  config.train.model_update_conf({"naive_conf": {}})
  return config

def mlp(data):
  initializer = flow.truncated_normal(0.1)
  reshape = flow.reshape(data, [data.shape[0], -1])
  hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer)
  return flow.layers.dense(hidden, 10, kernel_initializer=initializer)

@flow.function(get_train_config())
def train_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE, ), dtype=flow.int32)):
  with flow.fixed_placement("cpu", "0:0"):
    logits = mlp(images)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
  flow.losses.add_loss(loss)
  return loss


if __name__ == '__main__':
  check_point = flow.train.CheckPoint()
  check_point.init()
  (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE)
  for epoch in range(1):
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
      loss = train_job(images, labels).get().mean()
      if i % 20 == 0: print(loss)
```

## 代码说明
###  导入oneflow
```python
import oneflow as flow #导入oneflow
import numpy as np
from mnist_util import load_data
```

### 加载数据
通过oneflow内置函数load_data加载Mnist数据集。
```python
(train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE)
```
如果你是第一次使用，会自动下载mnist数据集文件至当前路径。
**output**:
```
====================== 38.0%|100%
```

### 训练环境配置及模型选择
**训练环境配置**
oneflow使用oneflow.function_config进行配置，在这里我们采取简单的默认配置。
```python
def get_train_config():
  config = flow.function_config()
  config.default_data_type(flow.float)
  config.train.primary_lr(0.1)
  config.train.model_update_conf({"naive_conf": {}})
  return config
```

**模型选择**
我们使用内置的oneflow.layers.dense方法构建mlp模型，并选择内置的sparse_softmax_cross_entropy_with_logits函数作为loss函数。
```python
def mlp(data):
  initializer = flow.truncated_normal(0.1)
  reshape = flow.reshape(data, [data.shape[0], -1])
  hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer)
  return flow.layers.dense(hidden, 10, kernel_initializer=initializer)

@flow.function(get_train_config())
def train_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE, ), dtype=flow.int32)):
  with flow.fixed_placement("cpu", "0:0"):
    logits = mlp(images)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
  flow.losses.add_loss(loss)
  return loss
```

###  开始训练
以上准备工作完成后，初始化网络，遍历数据集，就可以开始训练。
```python
  check_point = flow.train.CheckPoint()
  check_point.init()
  (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE)
  for epoch in range(1):
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
      loss = train_job(images, labels).get().mean()
      if i % 20 == 0: print(loss)
```
**output**：
```
2.7290366
0.81281316
0.50629824
0.35949975
0.35245502
...
```