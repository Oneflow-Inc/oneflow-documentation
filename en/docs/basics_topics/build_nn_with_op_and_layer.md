# Use OneFlow to build a neural network

The example in the previous section [ Recognition of MNIST handwritten numbers](../quick_start/lenet_mnist.md), we use the network layer in flow.Layers and operators in flow.nn to build a simple lenet network. Now we will use a simple neural network to introduce the core operator and layer in OneFlow.

The code in below is a nerural network which is constructed by convolution layer, pooling layer and fully connected layer. The figure shows the part and shape of the operator in network. The shape of `data` is 100x1×28×28. First of all `data` is the input of `conv2d` to take part in convolution, then we get the output `conv1`. After that conv1 will be the input of `max_pool2d` and keep run like this.(Note: Here isn't accurate, it is easy to understand such a description. We will explain it later)

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

The image havs two types of elements.One is the unit of operations represented by the box, including 'op' and 'layer', like `conv2d` 、 `dense` 、 `max_pool2d` and other.One is the data represented by arrows. 

## Operator and layer
Operator is a commonly used concept. It is the basic calculation unit in OneFlow. `reshape` and `nn.max_pool2d` is two types of operator. 
`layers.conv2d` and `layers.dense` is not operator. They are layers which constructed by special operators.For example, `layers.conv2d` is constructed by `conv2d` operator and `variable` operator. Layer can simplify the mapping process in front end. For more details, you can refer to: [layers api](https://oneflow-api.readthedocs.io/en/latest/layers.html).

## Blocks of data during network building
As mentioned above, it is inaccurate to say that "the parameter 'data' is the data with a dimension of '100x1×28×28'".Actually, when defining network, there is no data in 'data', it just a placeholder.

The network constructing process and running process is separated in fact. The build process is OneFlow underlying calculation chart based on the description of the network connection relation and the process, but the real calculation happened at run time.

When we build the network, we only describe the attributes and shapes(such as `shape`, `dtype`) of the nodes in network.There is no value in node, we call the node as **PlaceHolder**, OneFlow can compile and infer based on these placeholders to get the final computed graph. 

In OneFlow language the placeholder usually called `Blob`. There is a corresponding base clas `BlobDef` in OneFlow.

When building network, could print of the information of `Blob`. For example, we can print data `shape` and `dtype` as follow.
```
print(conv1.shape, conv1.dtype)
```

### Operator overloading
Operator overloading is defined in `BlobDef` which means object of `BlobDef` could handle add, subtract, multiply ,divide and some other operations.

Like '+' in the following code:

```
output = output + fc2_biases
```
Same as:
```
output = flow.broadcast_add(output, fc2_biases)
```

## Summary
Use OneFlow to construct the neural network need OneFlow to provide operator or layer as calculation unit. `BlobDef` acts as input and output to operator and layer, and operator overloading helps simplify some statements in code.
