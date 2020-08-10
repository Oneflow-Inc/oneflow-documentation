# How to obtain data from middle layer

OneFlow support `oneflow.watch` and `oneflow.watch_diff`. We can use them to register callback function to get data or gradient tensor in job functions at run time.

## Using guidance

To get data or gradient tensor in job function, we need to follow these steps:

* Write a callback function and the parameters of the callback function should be annotated to indicate the data type. The logic of callback function need to be config by user themselves.

* When we define the job functions, we use `oneflow.watch` or `oneflow.watch_diff` to register callback function. By the former one we can obtain the data tensor and by the latter one we get corresponding gradient.

* OneFlow will call the callback function at appropriate time at run time.

Take `oneflow.watch` as example:

```python
def MyWatch(x: T):
    #process x

@global_function()
def foo() -> T:
    #define network ...
    oneflow.watch(x, MyWatch)
    #...
```

The T in above code is the data type in `oneflow.typing`. Like  `oneflow.typing.Numpy`. More details please refer to [The definition and call of job function](job_function_define_call.md)

We will use the following examples to demonstrate how to use  `watch` and `watch_diff`.

## Use watch to obtain the data when running

The following is an example too demonstrate how to use `oneflow.watch` to obtain the data from middle layer in OneFlow.
```python
# test_watch.py
import numpy as np
import oneflow as flow
import oneflow.typing as tp


def watch_handler(y: tp.Numpy):
    print("out:", y)


@flow.global_function()
def ReluJob(x: tp.Numpy.Placeholder((5,))) -> None:
    y = flow.nn.relu(x)
    flow.watch(y, watch_handler)


flow.config.gpu_device_num(1)
data = np.random.uniform(-1, 1, 5).astype(np.float32)
print("in:", data)
ReluJob(data)
```

Run [above code](../code/extended_topics/test_watch.py):
```
python3 test_watch.py
```

We can get results like followings:
```
in: [ 0.15727027  0.45887455  0.10939325  0.66666406 -0.62354755]
out: [0.15727027 0.45887455 0.10939325 0.66666406 0.        ]
```

### Code explanation
In the example, we focus on `y` in `ReluJob`. Thus, we call `flow.watch(y, watch_handler)` to monitor `y`. The function `oneflow.watch` need two parameters:

* The first parameter is y which we are focus on.

* The second parameter is a callback function. When OneFlow use device resources to execute `ReluJob`, it will send `y` as a parameter to callback function. We define our callback function  `watch_handler` to print out its parameters.

User can use customized callback function to process the data from OneFlow according to their own requirements.

## Use watch_diff to obtain gradient when running
### `test_watch_diff.py`
The following is an example to demonstrate how to use `oneflow.watch_diff` to obtain the gradient at run time.
```python
# test_watch_diff.py
import oneflow as flow
import oneflow.typing as tp

BATCH_SIZE = 100


def watch_diff_handler(blob: tp.Numpy):
    print("watch_diff_handler:", blob, blob.shape, blob.dtype)


def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config


@flow.global_function(type="train", function_config=get_train_config())
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
    with flow.scope.placement("cpu", "0:0"):
        initializer = flow.truncated_normal(0.1)
        reshape = flow.reshape(images, [images.shape[0], -1])
        hidden = flow.layers.dense(
            reshape,
            512,
            activation=flow.nn.relu,
            kernel_initializer=initializer,
            name="hidden",
        )
        logits = flow.layers.dense(
            hidden, 10, kernel_initializer=initializer, name="output"
        )
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    flow.watch_diff(logits, watch_diff_handler)
    return loss


if __name__ == "__main__":
    check_point = flow.train.CheckPoint()
    check_point.init()

    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE
    )
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
        loss = train_job(images, labels)
        if i % 20 == 0:
            print(loss.mean())
```

Run [above code](../code/extended_topics/test_watch_diff.py):
```
python3 test_watch.py
```
We should have the following results:
```text
[ ...
 [ 1.39966095e-03  3.49164731e-03  3.31605263e-02  4.50417027e-03
   7.73609674e-04  4.89911772e-02  2.47627571e-02  7.65468649e-05
  -1.18361652e-01  1.20161276e-03]] (100, 10) float32
```
### Code explanation
In the example above, we use `oneflow.watch_diff` to obtain the gradient. The processe is same as the example which using `oneflow.watch`  to obtain data tensor.

First, we define the callback function:
```python
def watch_diff_handler(blob: tp.Numpy):
    print("watch_diff_handler:", blob, blob.shape, blob.dtype)
```

Then we use `oneflow.watch_diff` to register the callback function in job function:
```python
flow.watch_diff(logits, watch_diff_handler)
```

When running, OneFlow framework will call `watch_diff_handler` and send the gradient corresponding to `logits` above to `watch_diff_handler`.
