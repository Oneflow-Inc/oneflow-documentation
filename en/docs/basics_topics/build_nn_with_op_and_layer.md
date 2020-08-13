# Build a Neural Network

In the article [Recognition of MNIST Handwritten Digits](../quick_start/lenet_mnist.md), we have used "operator" in `oneflow.nn` and "layer" in `oneflow.layers` to build a LeNet neural network. Now we will use this simple neural network to introduce the core element for network construction in OneFlow: operator and layer.

LeNet is constructed by convolution layer, pooling layer and fully connected layer. 

<div align="center">
<img src="imgs/lenet.png" align='center'/>
</div>

The figure shows the inputs, outputs and layers in the network. There are two types of elements : one is calculation which represented by the box, including "op" and "layer", like `conv2d`, `dense`, `max_pool2d` and so on; the other is data which represented by arrows.

The input of the network is `data` and it's shape is 100x1×28×28. `conv2d` takes `data` as input and produce output, then this output is used as input for `conv1`. After that the output of `conv1` is passed as input to `max_pool2d`, and so on.

(Note: The input and output here is just place holder. We will explain this later.)

```python
def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(
        data,
        32,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        name="conv1",
        kernel_initializer=initializer,
    )
    pool1 = flow.nn.max_pool2d(
        conv1, ksize=2, strides=2, padding="SAME", name="pool1", data_format="NCHW"
    )
    conv2 = flow.layers.conv2d(
        pool1,
        64,
        5,
        padding="SAME",
        activation=flow.nn.relu,
        name="conv2",
        kernel_initializer=initializer,
    )
    pool2 = flow.nn.max_pool2d(
        conv2, ksize=2, strides=2, padding="SAME", name="pool2", data_format="NCHW"
    )
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(
        reshape,
        512,
        activation=flow.nn.relu,
        kernel_initializer=initializer,
        name="dense1",
    )
    if train:
        hidden = flow.nn.dropout(hidden, rate=0.5, name="dropout")
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="dense2")
```

## Operator and Layer
Operator is a common concept. It is the basic calculation unit in OneFlow. `reshape` and `nn.max_pool2d` used in LeNet code are two kinds of operators.

In contrast, `layers.conv2d` and `layers.dense` are not operator. They are layers which constructed by specific operators. For example, `layers.conv2d` is constructed by `conv2d` operator and `variable` operator. Layer can simplify the network construction procedure in the front end. For more details, please refer to [layers api](https://oneflow-api.readthedocs.io/en/latest/layers.html).

## Data block in neural network
As mentioned above, it is inaccurate to say that `data` is used as the input. Actually, when we define a network, there is no actual data in data blob, it's just a placeholder.

 The neural net construction and running process in OneFlow are actually separate. The construction process is a process in which OneFlow builds the computation graph according to the description defined with job function, but the real calculation happens at run time.

When building the network by defining job function, we only describe the attributes and shapes(such as `shape`, `dtype`) of the nodes in network. There is no data in the node, we call the node as **PlaceHolder**, OneFlow can compile and infer according to these placeholders to get the computation graph. 

The placeholders are usually called `Blob` in OneFlow. There is a corresponding base class `BlobDef` in OneFlow.

When we build network, we can print the information of `Blob`. For example, we can print data `shape` and `dtype` as follow.
```python
print(conv1.shape, conv1.dtype)
```

### Operator Overloading
The `BlobDef` class implements operator overloading which means `BlobDef` supports math operators such as addition, subtraction, multiplication and division and so on.

Like '+' in the following code:

```
output = output + fc2_biases
```
which is same as:
```
output = flow.broadcast_add(output, fc2_biases)
```

## Summary
When we build neural network, there are "operator" and "layer" provided by OneFlow as calculation units. "operator" and "layer" take `BlobDef` as input and output. Operator overloading on blob simplifies coding at python frontend.
