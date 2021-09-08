# Build Neural Network

The layers of a neural network can be built by API in namespace [oneflow.nn](https://oneflow.readthedocs.io/en/master/nn.html), It provides common Module (such as [oneflow.nn.Conv2d](https://oneflow.readthedocs.io/en/master/nn.html?highlight=oneflow.nn.Conv2D#oneflow.nn.Conv2d), [oneflow.nn.ReLU](https://oneflow.readthedocs.io/en/master/nn.html?highlight=oneflow.nn.ReLU#oneflow.nn.ReLU)). All Module classes inherit from [oneflow.nn.Module](https://oneflow.readthedocs.io/en/master/module.html#oneflow.nn.Module), and many simple Module can form more complex Module. In this way, users can easily build and manage complex neural networks.

```python
import oneflow as flow
import oneflow.nn as nn
```


## Define Module class

`oneflow.nn` provides common Module classes and we can use them easily. Or we can build a neural network by customizing the Module class on the basis. This method consists of three parts：

- Write a class that inherits from `oneflow.nn.Module` class
- Write the method of `__init__` class, in which we construct the network structure 
- Write the method of `forward` class, which calculates on the basis of the input of Module

```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
net = NeuralNetwork()
print(net)
```

The above code will output the structure of the `NeuralNetwork` network：

```text
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
    (5): ReLU()
  )
)
```

Then, call `net` (notice：It is not recommended to explicitly call `forward`):

```python
X = flow.ones(1, 28, 28)
logits = net(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```

You will get output similar to the following:

```text
Predicted class: tensor([1], dtype=oneflow.int32)
```

The above process of data input, network calculation and the output of reasoning is shown in the figure below:

![todo](./imgs/neural-network-layers.png)

## `flow.nn.functional`

In addition to `oneflow.nn`, [oneflow.nn.functional](https://oneflow.readthedocs.io/en/master/functional.html) namespace also provides many API. It overlaps with `oneflow.nn` to some extent. For example, [nn.functional.relu](https://oneflow.readthedocs.io/en/master/functional.html?highlight=relu#oneflow.nn.functional.relu) and [nn.ReLU](https://oneflow.readthedocs.io/en/master/nn.html?highlight=relu#oneflow.nn.ReLU) both can be used for activation in neural network.

The main differences between them are:

- The API under `nn` is a class. It needs to be instantiated before being called; The API under `nn.functional` is a function. It is called directly.
- The API under `nn` manages network parameters automatically；But for the function under `NN. Functional`, we need to define our own parameters and manually pass them in each call.

In fact, most of the Module provided by OneFlow is the result of encapsulating the methods under `nn.functional`. `nn.functional` can manage the network more finely.

The following example uses the methods in `nn.functional` to build a Module `FunctionalNeuralNetwork` equivalent to the `NeuralNetwork` class above. Readers can appreciate the similarities and differences between the two:

```python
class FunctionalNeuralNetwork(nn.Module):    
    def __init__(self):
        super(FunctionalNeuralNetwork, self).__init__()
        
        self.weight1 = nn.Parameter(flow.randn(28*28, 512))
        self.bias1 = nn.Parameter(flow.randn(512))

        self.weight2 = nn.Parameter(flow.randn(512, 512))
        self.bias2 = nn.Parameter(flow.randn(512))

        self.weight3 = nn.Parameter(flow.randn(512, 10))
        self.bias3 = nn.Parameter(flow.randn(10))
        
    def forward(self, x):
        x = x.reshape(1, 28*28)
        out = flow.matmul(x, self.weight1)
        out = out + self.bias1
        out = nn.functional.relu(out)

        out = flow.matmul(out, self.weight2)
        out = out + self.bias2
        out = nn.functional.relu(out)

        out = flow.matmul(out, self.weight3)
        out = out + self.bias3
        out = nn.functional.relu(out)

        return out

net = FunctionalNeuralNetwork()
X = flow.ones(1, 28, 28)
logits = net(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```


## Module 容器

比较以上 `NeuralNetwork` 与 `FunctionalNeuralNetwork` 实现的异同，可以发现 [nn.Sequential](https://oneflow.readthedocs.io/en/master/nn.html?highlight=nn.Sequential#oneflow.nn.Sequential) 对于简化代码起到了重要作用。

`nn.Sequential` 是一种特殊容器，只要是继承自 `nn.Module` 的类都可以放置放置到其中。

它的特殊之处在于：当 Sequential 进行前向传播时，Sequential 会自动地将容器中包含的各层“串联”起来。具体来说，会按照各层加入 Sequential 的顺序，自动地将上一层的输出，作为下一层的输入传递，直到得到整个 Moudle 的最后一层的输出。

以下是不使用 Sequential 构建网络的例子（不推荐）：

```python
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1,20,5)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(20,64,5)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        return out
```

如果使用 Sequential，则看起来是这样，会显得更简洁。

```python
class MySeqModel(nn.Module):
    def __init__(self):
        super(MySeqModel, self).__init__()
        self.seq = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )

    def forward(self, x):
        return self.seq(x)
```

除了 Sequential 外，还有 `nn.Modulelist` 及 `nn.ModuleDict`，除了会自动注册参数到整个网络外，他们的其它行为类似 Python list、Python dict，只是常用简单的容器，不会自动进行前后层的前向传播，需要自己手工遍历完成各层的计算。 
