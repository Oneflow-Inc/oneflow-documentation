
我们可以使用`oneflow.get_variable`方法创造或者获取一个blob类型的变量，该变量可以用于在全局任务函数中交互信息。
因为这个特点，常常使用`get_variable`获取对象，用于存储模型参数。

## get_variable获取/创建对象的流程

`get_variable`需要一个指定一个`name`参数，该参数作为创建对象的标识。
如果`name`指定的值在当前上下文环境中已经存在，那么get_variable会取出已有的blob对象，并返回。

如果`name`指定的值不存在，则get_varialbe内部会创建一个blob对象，并返回。

## 使用get_variable创建对象

`oneflow.get_variable`的原型如下：

```python
def get_variable(
    name,
    shape=None,
    dtype=None,
    initializer=None,
    regularizer=None,
    trainable=None,
    model_name=None,
    random_seed=None,
    distribute=distribute_util.broadcast(),
)
```

以下是`oneflow.layers.conv2d`中，使用get_variable创造参数变量，并进一步构建网络的例子：

```python
    #...
    weight = flow.get_variable(
        weight_name if weight_name else name_prefix + "-weight",
        shape=weight_shape,
        dtype=inputs.dtype,
        initializer=kernel_initializer
        if kernel_initializer is not None
        else flow.constant_initializer(0),
        regularizer=kernel_regularizer,
        trainable=trainable,
        model_name="weight",
    )

    output = flow.nn.conv2d(
        inputs, weight, strides, padding, data_format, dilation_rate, groups=groups, name=name
    )
    #...
```
