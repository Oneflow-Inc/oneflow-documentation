# The Definition and Call of Job Function

In OneFlow, we encapsulate the training, inference and some other tasks into a "job function". The job function is used to connect the user's business logic and the computing resource managed by OneFlow. 

In OneFlow, the function decorated by `@oneflow.global_function` decorator is the OneFlow's job function

We mainly define the structure of the model and choose the optimization target in job function. In addition, we can also pass some hyperparameters about training and some configuration of the environment to the job function (like the following example: `get_train_config()`), OneFlow will manage the memory, GPU and other computing resource according to our configuration.

In this article, we will specifically learn about:

* how to define and call the job function

* how to get the return value of job function

## The Relationship Between Job Function and Running Process of OneFlow

The job function is divided into two phases: definition and call.

It's related to OneFlow's operating mechanism. Briefly, the OneFlow Python layer API simply describes the configuration and the training environment of the model. These information will pass to the C++ backend. After compilation, graph building and so on, the computation graph is obtained. Finally, the job function will be executed in OneFlow runtime.

The job function describes the model and the training environment. In this phase there's no data. We can only define the shape and data type of the nodes (as known as **PlaceHolder**) for creating and compiling the computation graph of OneFlow.

The job function will be called after the OneFlow runtime starts. We can pass the data by calling job function and get the results. 

We will introduce the definition and calling method of job functions in detail as below. 

## The Definition of Job Function

We encapsulate the model in Python and use `oneflow.global_function` to decorate. Then the definition is completed.

The job function mainly describes two things:

* The structure of model

* The optimizing target in training phase

In the following code, we build a Multi-Layer Perceptron model and use `flow.nn.sparse_softmax_cross_entropy_with_logits` to compute the cross-entropy loss as our optimizing target.

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

### The Parameters of `oneflow.global_function`

`oneflow.global_function` decorator accepts two parameters, `type` and `function_config`. 

- The parameter `type` accepts a string, which can only set as `train` or `predict`. When we define a training model, we set it as `train`. We set is as `predict` when we define a model for testing or inferencing. 
- The parameter `function_config` accepts an object which is constructed by `oneflow.function_config()`. In `function_config` object, we can use its method or attribute to config. As the following code. 

```python
def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config
```

We set the default data type, then, we can pass the `function_config` object to the `global_function` decorator. 

```python
@flow.global_function(type="train", function_config=get_train_config())
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
```

For the complete code, you can refer to [Consistent and Mirrored](consistent_mirrored.md)'s  [mixed_parallel_mlp.py](../code/extended_topics/hybrid_parallelism_mlp.py)

### PlaceHolder

Noted that the `images`、`logits`、`labels`、`loss` and some other objects have no data in our definition of the job function. They are used to describe **the shape and attribute of data**, which is called **PlaceHolder**.

For the parameters of the job function, we use `Numpy.Placeholder`, `ListNumpy.Placeholder`, `ListListNumpy.Placeholder` in the `oneflow.typing` package to annotate the data type of them as `numpy.ndarray`, `Sequence[numpy.ndarray]` and `Sequence[Sequence[numpy.ndarray]]` respectively.

Besides the types of `oneflow.typing`, the variables returned from OneFlow operators or layers in the job function, like the `reshape`、`hidden`、`logits`、`loss` in the code above, are also PlaceHolder.

All the variables mentioned above inherit the base class `BlobDef` directly or indirectly. We call this object type as **Blob** in OneFlow. 

The **Blob** has no data when defining the job function. It only plays the role of data placeholder for building the graph.

### The Return Value of the Job Function

The concept of the data placeholder **Blob** is emphasized above because the return value of the job function cannot be arbitrarily specified. It must be `Blob` type object or a container which only contains the `Blob` object. 

For example, the `loss` returned in the above code is a `Blob` object

The return values of job function should be annotated. As an example, `-> tp.Numpy` in above code means the function returns a `Blob` object.

As another example, we can annotate the return value type as `-> Tuple[tp.Numpy, tp.Numpy]`.It means the function returns a `tuple` which contains two `Blob` object

You can refer to [Get the result of the job function](../basics_topics/async_get.md) for specific examples.

## The Call of Job Function

OneFlow uses decorator to convert Python function into OneFlow's job function. It is transparent to user.

We can call the job function just like we call a Python function. Every time we call the job function, OneFlow will complete the forward propagation, back propagation, parameter updates, and more in framework. 

In the code below. When we get the data, we will pass parameters and call the `train_job` function to print `loss`. 

```python
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE
    )

    for epoch in range(3):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            loss = train_job(images, labels)
            if i % 20 == 0:
                print(loss.mean())
```

As you can see, by calling the job function `train_job`, the `numpy` data is directly returned.

The method shown above is synchronous. OneFlow also supports asynchronous invocation. For more details you can refer to the article [Get the result of the job function](../basics_topics/async_get.md).

