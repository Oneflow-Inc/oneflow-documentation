# Build a Neural Network

In the article [Recognition of MNIST Handwritten Digits](../quick_start/lenet_mnist.md), we use the "layer" and "operators" in `oneflow.layers` and `oneflow.nn` to build a simple LeNet network. Now we will use a simple neural network to introduce the core of network construction in OneFlow: operator and layer.

The code below shows a nerural network which is constructed by convolution layer, pooling layer and fully connected layer. The figure shows the input and output of operator in the network. The shape of `data` is 100x1×28×28. 

Firstly, `data` is used as the input of `conv2d` to the convolution calculation, and the output `conv1` is obtained. 

After that, `conv1` is passed as input to `max_pool2d`, and so on. 

(Note: the description here is a little not accurate. It is only convenient to understand and will be explained later)

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

<div align="center">
<img src="imgs/Lenet.png" align='center'/>
</div>

There are two types of elements in this figure above. One is calculating unit which represented by the box, including "op" and "layer", like `conv2d`, `dense`, `max_pool2d` and so on. The other is the data represented by arrows. 

## Operator and layer
Operator is a common concept. It is the basic calculation unit in OneFlow. `reshape` and `nn.max_pool2d` are two kinds of operators.

In contrast, `layers.conv2d` and `layers.dense` are not operator. They are layers which constructed by specific operators. For example, `layers.conv2d` is constructed by `conv2d` operator and `variable` operator. Layer can simplify the network construction procedure in the front end. For more details, you can refer to [layers api](https://oneflow-api.readthedocs.io/en/latest/layers.html).

## Data block in neural network
As mentioned above, it is inaccurate to say that "data is used as the input" in that figure. Actually, when we define a network, there is no data in 'data', it's just a placeholder.

 The construction process and running process of network in OneFlow are actually separate. The construction process is a process in which OneFlow builds the calculation graph according to the description of network defined by job function, but the real calculation happens at run time.

When we build the network by defining job function, we only describe the attributes and shapes(such as `shape`, `dtype`) of the nodes in network. There is no data in node, we call the node as **PlaceHolder**, OneFlow can compile and infer according to these placeholders to get the computation graph. 

The placeholders are usually called `Blob` in OneFlow. There is a corresponding base class `BlobDef` in OneFlow.

When we build network, we can print the information of `Blob`. For example, we can print data `shape` and `dtype` as follow.
```python
print(conv1.shape, conv1.dtype)
```

### Operator overloading
The `BlobDef` class implments operator overloading which means `BlobDef` supports operations such as addition, subtraction, multiplication and division and some other operations.

Like '+' in the following code:

```
output = output + fc2_biases
```
Same as:
```
output = flow.broadcast_add(output, fc2_biases)
```

## Summary
When we build the neural network, there are caculation unit "operator" and "layer"  provided by OneFlow as calculation unit. Operator and layer take `BlobDef` as input and output. The operator overlaoding of blob simplify some statements in code.
