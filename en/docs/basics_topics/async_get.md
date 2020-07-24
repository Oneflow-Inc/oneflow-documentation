# Obtain results from job function

In this article, we will mainly introduce how to obtain the return value from job function in OneFlow which includes:

* How to use synchronously method obtained the return value from job function.

* How to use asynchronous method obtained the return value from job function.

In OneFlow, we usually use the decorator called @flow.global_function to defined job function. Thus, This task could be training, evaluation or prediction.We can use `get()` and `async_get()` to obtain the return object from a job function. `get` and `async_get` is apply in corresponding job function.

## Difference between synchronous and asynchronous

Normally, our trainin process is synonymous which mean in line. Now we will demonstrated the concept of synchronous and asynchronous and the advantages of asynchronous in OneFlow by a simple example.

#### Synchronous

During the complete process in one iteration, when the data from some step/iter completed the forward and reverse transmission process and completed the updated of weight parameters and the optimizer. Then start the training process in next step.Whereas before next step, usually we need to wait for CPU prepare the training data. Normally it will come up with some times for data preprocessing and loading.

#### Asynchronous

During the iteration with asynchronous process, it basic means opens the multithreaded mode. One step does not need to wait for the previous step to finish, but it can directly process data preprocessing and loading. When GPU is not full loading, it can start training directly.When the GPU is full loading, then it can start preparing work for other step.

From the contrast mentioned above, OneFlow preform more efficient of using the computing resources when using the asynchronous mode. Especially when enormous amount of dataset are apply. **Open asynchronous process could narrow the time of data loading and data preparation and boost training model**.



Next, we will introduce how to obtain result in synchronous and asynchronous task and the coding of call function in asynchronous task. The complete source code will be provide in the end of page.

## Obtain result in synchronous

When calling the job function, we will get an OneFlow object. The method `get` of this object can get result by synchronous.

For example, we defined the function below:
```python
@flow.global_function(get_train_config())
def train_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE,), dtype=flow.int32)):
    with flow.scope.placement("cpu", "0:0"):
        logits = lenet(images, train=True)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
    flow.losses.add_loss(loss)
    return loss
```

Thus, we can use the following code, use `get`  to obtain the return loss in the job function and print the average of them.

```python
loss = train_job(images, labels).get()
print(loss.mean())
```

From the example above, we should notice that:

Because of the characteristic of OneFlow frame work, the `return `object when define the function **can not** directly get when call the job function. It need use `get` (next chapter will introduce`async_get` ).


## Obtain result in asynchronous

Normally, the efficiency of asynchronous is better than synchronous. The following is introduced how to use `async_get` to obtain the result from a job function which is asynchronous training.

Basic steps include:

* Prepare callback function and achieve return the result of logic in  function of processing.

* Use async_get to regist callback.

* OneFlow find the suitable time to regist the callback and job function return the result to the callback.

The first two step is done by the user and the final step is done by OneFlow framework.

### Coding of callback function
Prototype of callback function:

```python
def cb_func(result):
    #...
```

The result is the return value of job function.

For example, in the job function below, the return is loss.

```python
@flow.global_function(get_train_config())
def train_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE, ), dtype=flow.int32)):
  #mlp
  #... code not shown
  logits = flow.layers.dense(hidden, 10, kernel_initializer=initializer)

  loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

  flow.losses.add_loss(loss)
  return loss
```

Corresponding callback function, just print the average of loss:

```python
g_i = 0
def cb_print_loss(result):
  global g_i
  if g_i % 20 == 0:
    print(result.mean())
  g_i+=1
```

Another example, the job function below:

```python
@flow.global_function(get_eval_config())
def eval_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE, ), dtype=flow.int32)):
  with flow.scope.placement("gpu", "0:0"):
    logits = lenet(images, train=True)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

  return {"labels":labels, "logits":logits}
```

The returen object is a dictionary and it store two elements which is labels and logits. We can use the callback function below, handle both of them and calculate the accuracy:

```python
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

### Registration of callback function
When call the job function, will return object `blob`. Call `async_get` in that object. It can regist the callback function we already prepared.

```python
train_job(images,labels).async_get(cb_print_loss)
```

OneFlow automatically call the registed callback function when obtain the training result.


## The relevant code

### Synchronised obtain a result
In this example, use a simple Multilayer perceptron(mlp), use synchronization to obtain the only return value `loss` and print the average of loss in each 20 iterations.

Name：[synchronize_single_job.py](../code/basics_topics/synchronize_single_job.py)

```python
import oneflow as flow
from mnist_util import load_data

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
    config.default_data_type(flow.float)
    config.train.primary_lr(0.1)
    config.train.model_update_conf({"naive_conf": {}})
    return config


@flow.global_function(get_train_config())
def train_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE,), dtype=flow.int32)):
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=True)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
    flow.losses.add_loss(loss)
    return loss


if __name__ == '__main__':

    flow.config.enable_debug_mode(True)
    check_point = flow.train.CheckPoint()
    check_point.init()
    (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE)
    for epoch in range(1):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            loss = train_job(images, labels).get().mean()
            if i % 20 == 0: print(loss)
    check_point.save('./lenet_models_1')  # need remove the existed folder
```

### Synchronised obtain multiple results
In this example, the return object of job function is a `list `. We can use synchronization to obtain the elements like `labels` and `logits` in the `list`. And evaluate the model we trained before then print the accuracy.

Name：[synchronize_batch_job.py](../code/basics_topics/synchronize_batch_job.py)

```python
import numpy as np
import oneflow as flow
from mnist_util import load_data

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


def get_eval_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config


@flow.global_function(get_eval_config())
def eval_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
             labels=flow.FixedTensorDef((BATCH_SIZE,), dtype=flow.int32)):
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=True)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

    return [labels, logits]


g_total = 0
g_correct = 0


def acc(labels, logits):
    global g_total
    global g_correct

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
            labels, logits = eval_job(images, labels).get()
            acc(labels, logits)

    print("accuracy: {0:.1f}%".format(g_correct * 100 / g_total))
```

The model have already trained can be downloaded in: [lenet_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/lenet_models_1.zip)

### Asynchronously obtain a result

In this example, using mlp training,  obtain the only return value `loss` by asynchronous way and print the average of loss in each 20 times of iterations.

Name：[async_single_job.py](../code/basics_topics/async_single_job.py)

```python
import oneflow as flow
from mnist_util import load_data

BATCH_SIZE = 100


def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    config.train.primary_lr(0.1)
    config.train.model_update_conf({"naive_conf": {}})
    return config


@flow.global_function(get_train_config())
def train_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE,), dtype=flow.int32)):
    # mlp
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(images, [images.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer)
    logits = flow.layers.dense(hidden, 10, kernel_initializer=initializer)

    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

    flow.losses.add_loss(loss)
    return loss


g_i = 0


def cb_print_loss(result):
    global g_i
    if g_i % 20 == 0:
        print(result.mean())
    g_i += 1


def main_train():
    # flow.config.enable_debug_mode(True)
    check_point = flow.train.CheckPoint()
    check_point.init()

    (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE)
    for epoch in range(50):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            train_job(images, labels).async_get(cb_print_loss)

    check_point.save('./mlp_models_1')  # need remove the existed folder


if __name__ == '__main__':
    main_train()
```

The model have already trained can be downloaded in: [mlp_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/mlp_models_1.zip)

### Asynchronously obtain multiple results

In this example, the return object of job function is a `dict `. We can use asynchronization to obtain multiple return objects in `dict`. And evaluate the model we trained before then print the accuracy.

Name：[async_batch_job.py](../code/basics_topics/async_batch_job.py)

```python
import numpy as np
import oneflow as flow
from mnist_util import load_data

BATCH_SIZE = 100


def mlp(data):
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(data, [data.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer)
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer)


def get_eval_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config


@flow.global_function(get_eval_config())
def eval_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
             labels=flow.FixedTensorDef((BATCH_SIZE,), dtype=flow.int32)):
    with flow.scope.placement("cpu", "0:0"):
        logits = mlp(images)
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


def main_eval():
    # flow.config.enable_debug_mode(True)
    check_point = flow.train.CheckPoint()
    check_point.load('./mlp_models_1')
    (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE)
    for epoch in range(1):
        for i, (images, labels) in enumerate(zip(test_images, test_labels)):
            eval_job(images, labels).async_get(acc)

    print("accuracy: {0:.1f}%".format(g_correct * 100 / g_total))


if __name__ == '__main__':
    main_eval()
```
