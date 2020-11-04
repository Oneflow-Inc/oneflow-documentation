# How to Obtain Data from Middle Layer

OneFlow support `oneflow.watch` and `oneflow.watch_diff`. We can use them to register a callback function to get data and gradient tensor in job functions at runtime.

## Using Guidance

To get data or gradient tensor in job function, we need to follow these steps:

* Write a callback function and the parameters of the callback function should be annotated to indicate the data type. The logic of the callback function need to be set up by user themselves.

* When defining the job functions, we use `oneflow.watch` or `oneflow.watch_diff` to register callback function. We obtain data tensor from the former one and their corresponding gradient from the latter one.

* OneFlow will call the callback function at an appropriate time at runtime.

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

The T in the code above is the data type in `oneflow.typing`. Like  `oneflow.typing.Numpy`. Please refer to [The definition and call of job function](job_function_define_call.md) for more details.

We will use the following examples to demonstrate how to use  `watch` and `watch_diff`.

## Use watch to Obtain the Data when Running

The following is an example to demonstrate how to use `oneflow.watch` to obtain the data from middle layer in OneFlow.
Code:[test_watch.py](../code/extended_topics/test_watch.py)

Run above code:
```
python3 test_watch.py
```

We can get results like the followings:
```
in: [ 0.15727027  0.45887455  0.10939325  0.66666406 -0.62354755]
out: [0.15727027 0.45887455 0.10939325 0.66666406 0.        ]
```

### Code Explanation
In the example, we focus on `y` in `ReluJob`. Thus, we call `flow.watch(y, watch_handler)` to monitor `y`. The function `oneflow.watch` needs two parameters:

* The first parameter is y which we focus on.

* The second parameter is a callback function. When OneFlow use device resources to execute `ReluJob`, it will send `y` as a parameter to callback function. We define our callback function  `watch_handler` to print out its parameters.

User can use customized callback function to process the data from OneFlow according to their own requirements.

## Use watch_diff to Obtain Gradient when Running

The following is an example to demonstrate how to use `oneflow.watch_diff` to obtain the gradient at runtime.

Code: [test_watch_diff.py](../code/extended_topics/test_watch_diff.py)

Run above code:
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
### Code Explanation
In the example above, we use `oneflow.watch_diff` to obtain the gradient. The processe is the same as the example which using `oneflow.watch`  to obtain data tensor.

First, we define the callback function:
```python
def watch_diff_handler(blob: tp.Numpy):
    print("watch_diff_handler:", blob, blob.shape, blob.dtype)
```

Then we use `oneflow.watch_diff` to register the callback function in job function:
```python
flow.watch_diff(logits, watch_diff_handler)
```

When running, OneFlow framework will call `watch_diff_handler` and send the gradient corresponding with `logits` to `watch_diff_handler`.
