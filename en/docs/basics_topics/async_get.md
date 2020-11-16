# Get Results from Job Function

In this article, we will talk about getting the return value of a job function in OneFlow. It covers:

* How to get the return value from a job function synchronously.

* How to get the return value from a job function asynchronously.

In OneFlow, a function decorated by `@flow.global_function` is called "Job Function". Job function can be implmented for training, evaluation and prediction. By specifing the return type of job function, we can get results from job function both synchronously and asynchronously.

## Difference Between Synchronous and Asynchronous

### Synchronization

During synchronous training, the training of the next step cannot be started until the work of the previous step is completed.

### Asynchronization

In asynchronous training, it is equivalent to turning on multi-threading mode. A step does not have to wait until previous step completes. For instance data preprocessing and loading task could be runned in advance.

Through the comparison above, it can be seen that the use of asynchronous execution job function in OneFlow can effectively utilize computer resources, especially in the case of loading huge data, enabling asynchronous execution can effectively shorten the data loading and preprocessing time, and accelerate the model training.

Next, we will explain how to get the results synchronously and asynchronously from job function, and how to write callback functions in asynchronous jobs. At the end of the article, we will provide a complete example.

The main points are：

* Getting results synchronously or asynchronously is determined by the return value type of job function

* The data type of return value is selected in  `oneflow.typing`

* When we call a job function, the form of getting results synchronously / asynchronously is slightly different


## Get Result Synchronously

When we define a job function, if the annotation of the return type of the job function is `oneflow.typing.Numpy`, the job function will be called synchronously.

For example, when we define a job function like this:
```python
@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
    with flow.scope.placement("gpu", "0:0"):
        logits = lenet(images, train=True)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, logits, name="softmax_loss"
        )
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    return loss
```

Through Python annotations, OneFlow knows the type of the job function's return is `oneflow.typing.Numpy`, which corresponds to `ndarray` in `Numpy`.
Then when we call the job function, it will simply return the `ndarray` object:

```python
loss = train_job(images, labels)
if i % 20 == 0:
    print(loss.mean())
```

From the example above, it should be noted that:

* When we define the job function, the return value of job function (loss) is a placeholder for graph construction.

* When we specify the return value type as `oneflow.typing.Numpy`. OneFlow will know that when the job function is called, the real data type returned is `Numpy.ndarray` object.

* By calling the job function `train_job(images, labels)`, we can get the result from job function directly, and the retnred value is a `ndarray` object corresponding to `oneflow.typing.Numpy`.

## Data Types in `oneflow.typing`
The `oneflow.typing` contains all the data types that can be returned by the job function. The `oneflow.typing.Numpy` shown above is only one of them. The commonly used types and their corresponding meanings are listed as follows:

* `oneflow.typing.Numpy`: corresponding to a `numpy.ndarray`

* `flow.typing.ListNumpy`: corresponding to a `list` container. Each element in it is a `numpy.ndarray` object. It is related to the view of OneFlow for distributed training. We will see details in [The consistent and mirrored view in distributed training.](../extended_topics/consistent_mirrored.md)

* `oneflow.typing.ListListNumpy`：corresponds to a `list` container where each element is a `TensorList` object and some interfaces to OneFlow need to process or return multiple `TensorList`. More information refer to [Term & Concept in OneFlow](. /concept_explanation.md#3tensorbuffer-tensorlist) and related [API documentation](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight= ListListNumpy)

* `oneflow.typing.Callback`: corresponding to a callback function. It is used to call job function asynchronously. We will introduce it below.

In addition, OneFlow also allows job functions to pass out data in dictionary form. For examples of `ListNumpy`, `ListNumpy`, `ListListNumpy` and how to pass out data in dictionary form please refer to  [OneFlow's Test Case](https://github.com/ Oneflow-Inc/oneflow/blob/master/oneflow/python/test/ops/test_global_function_signature.py).

## Get Result Asynchronously

Generally speaking, the efficiency of asynchronous training is higher than that of synchronous training. The following describes how to call job function asynchronously and process the results.

The basic steps include:

* Prepare a callback function: It is necessary to specify the parameters accepted by the callback function by annotations and implementing the logic for return value of the job function inside the callback function.

* Implementing a Job Function: Specify `flow.typing.Callback` as the return type of the job function by annotation. As we will see in the following example that we can specify the parameter type of the callback function with `Callback`.

* Call the job function: Register the callback function prepared in step 1 above.

The first three steps are completed by users of OneFlow. When the program runs, the registered callback function is called by OneFlow and the return values of the job function are passed as parameters to the callback function.

### Prepare a Callback Function
Suppose the prototype of the callback function is:

```python
def cb_func(result:T):
    #...
```

Among them, the `result` needs to be annotated to specify its type `T`, that is, numpy, listnumpy, etc. mentioned above, or their composite type. We will have corresponding examples below.

The `result` of callback function is actually the return value of job function. Therefore, it must be consistent with the annotation of the return value of the job function

For example, when we define a job function below:
```python
@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Callback[tp.Numpy]:
    # mlp
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

    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
        labels, logits, name="softmax_loss"
    )
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    return loss
```

Annotation `-> tp.Callback[tp.Numpy]` means that this job function returns a `oneflow.typing.Numpy` type, and need to be called asynchronously.

Thus, the callback function we defined should accept a numpy parameter:
```python
def cb_print_loss(result: tp.Numpy):
    global g_i
    if g_i % 20 == 0:
        print(result.mean())
    g_i += 1
```

Let's take a look at another example. If the job function is defined as below:

```python
@flow.global_function(type="predict")
def eval_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Callback[Tuple[tp.Numpy, tp.Numpy]]:
    with flow.scope.placement("cpu", "0:0"):
        logits = mlp(images)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, logits, name="softmax_loss"
        )

    return (labels, logits)
```

Annotation `-> tp.Callback[Tuple[tp.Numpy, tp.Numpy]]` means that this job function returns a `tuple` and each element is `tp.Numpy`. The job function needs to be called asynchronously.

Thus, the parameter annotation of the corresponding callback function should be:

```python
g_total = 0
g_correct = 0


def acc(arguments: Tuple[tp.Numpy, tp.Numpy]):
    global g_total
    global g_correct

    labels = arguments[0]
    logits = arguments[1]
    predictions = np.argmax(logits, 1)
    right_count = np.sum(predictions == labels)
    g_total += labels.shape[0]
    g_correct += right_count
```
The `arguments` corresponds to the return type of the above job function.

### Registration of Callback Function
When we call the job function asynchronously, the job function will return a 'callback' object, and we register the prepared callback function by passing it to that object.

OneFlow will automatically call the registered callback function when it gets the training results.

```python
callbacker = train_job(images, labels)
callbacker(cb_print_loss)
```
But the above code is redundant, we recommend:

```python
train_job(images, labels)(cb_print_loss)
```

## Code

### Get single result synchronously
We use LeNet as an example here to show how to get the return value `loss` synchronously and print the loss every 20 iterations.

Code example: [synchronize_single_job.py](../code/basics_topics/synchronize_single_job.py)

Run:

```shell
wget https://docs.oneflow.org/code/basics_topics/synchronize_single_job.py
python3 synchronize_single_job.py
```

There will be outputs like:

```text
File mnist.npz already exist, path: ./mnist.npz
7.3258467
2.1435719
1.1712438
0.7531896
...
...
model saved
```

### 

### Get Mutiple Results Synchronously
In this case, the job function returns a `tuple`. We get the results `labels` and `logits` in tuple synchronously. Also, we evaluate the trained model in the above example, then output the accuracy rate.

Code: [synchronize_batch_job.py](../code/basics_topics/synchronize_batch_job.py)

The trained model can be downloaded from [lenet_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/lenet_models_1.zip)
Run:

```shell
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/quick_start/lenet_models_1.zip
unzip lenet_models_1.zip
wget https://docs.oneflow.org/code/basics_topics/synchronize_batch_job.py
python3 synchronize_batch_job.py
```

There will be outputs like:

```text
accuracy: 99.3%
```

### 

### Get Single Result Asynchronously

In this case, we take MLP as example to get the single return result loss from job function asynchronously, and the loss is printed every 20 rounds.

Code: [async_single_job.py](../code/basics_topics/async_single_job.py)

Run:

```shell
wget https://docs.oneflow.org/code/basics_topics/async_single_job.py
python3 async_single_job.py
```

There will be outputs like:

```text
File mnist.npz already exist, path: ./mnist.npz
3.0865736
0.8949808
0.47858357
0.3486296
...
```



### Get Multiple Results Asynchronously
In the following example, we will show how to get multiple return results of the job function asynchronously, evaluate the trained model in the above example, and get the accuracy.

Code: [async_batch_job.py](../code/basics_topics/async_batch_job.py)

The trained model can be downloaded from [mlp_models_1.zip](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/mlp_models_1.zip)

](https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/mlp_models_1.zip)

Run:

```shell
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/online_document/docs/basics_topics/mlp_models_1.zip
unzip mlp_models_1.zip
wget https://docs.oneflow.org/code/basics_topics/async_batch_job.py
python3 async_batch_job.py
```

There would be outputs like:

```text
File mnist.npz already exist, path: ./mnist.npz
accuracy: 97.6%
```

