# How to obtain data from middle layer

OneFlow support `oneflow.watch` and `oneflow.watch_diff`. We can use them to register callback function. In order to get multiple data or gradient tensor in job functions.

## Using guidance

To get multiple data or gradient tensor in job functions. We need do following step:

* Write a callback function and the parameters in callback function need use comment to indicate data type. The logic of callback function need to be config by user themselves.

* When define the job functions. Use  `oneflow.watch` or  `oneflow.watch_diff` to register callback function. <0>Oneflow.watch</0> obtain the data tensor and <0>oneflow.watch_diff</0> get corresponding gradient.

* When running the job function, OneFlow will call the logic in callback function in correct time.

Use `oneflow.watch` as example:

```python
def MyWatch(x: T):
    #process x

@global_function()
def foo() -> T:
    #define network ...
    oneflow.watch(x, MyWatch)
    #...
```

The T in above script is the data type in `oneflow.typing`. Like  `oneflow.typing.Numpy`. More details please reference to [Calling and definition of job function](job_function_define_call.md)

We will use the following examples to demonstrate how to use  `watch` and `watch_diff`.

## Use  `watch` to obtain the middle layer data when running

The following is an example. To demonstrate how to use `oneflow.watch` to obtain the data from middle layer in OneFlow.
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

Run [above script](../code/extended_topics/test_watch.py):
```
python3 test_watch.py
```

Should get following results:
```
in: [ 0.15727027  0.45887455  0.10939325  0.66666406 -0.62354755]
out: [0.15727027 0.45887455 0.10939325 0.66666406 0.        ]
```

### Script explanation
In examples, we focus on  `y` in `ReluJob`. Thus, we call `flow.watch(y, watch_handler)` to monitor `y`.`oneflow.watch` need two parameters:

* The first parameter is y which we are focus on.

* The second parameter is callback function. When OneFlow use device resources to execute `ReluJob`. It will send `y`  as a parameter to callback function.So we define our callback function  `watch_handler` to print out parameters.

User use custom callback function to process the data from OneFlow according their own requirements.

## Use  `watch_diff` to obtain gradient when running
### `test_watch_diff.py`
The following is an example. To demonstrate how to use `oneflow.watch_diff` to obtain the data from middle layer in OneFlow.
```python
# test_watch_diff.py
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
```

Run [above script](../code/extended_topics/test_watch_diff.py):
```
python3 test_watch.py
```
We should have the following results:
```
# test_watch_diff.py
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
```
### Script explanation
By the example above, we use `oneflow.watch_diff` to obtain the gradient. Other processes is same as the example which using `oneflow.watch`  to obtain data tensor.

First, define the callback function:
```python
def watch_diff_handler(blob: tp.Numpy):
    print("watch_diff_handler:", blob, blob.shape, blob.dtype)
```

Then use  `oneflow.watch_diff` to register the callback function in job function:
```python
flow.watch_diff(logits, watch_diff_handler)
```

When OneFlow running, OneFlow framework will call `watch_diff_handler` and send the gradient corresonding to `logits` above to  `watch_diff_handler`.
