# 搭建神经网络

​神经网络的各层，可以使用 [oneflow.nn](https://oneflow.readthedocs.io/en/master/nn.html) 名称空间下的 API 搭建，它提供了构建神经网络所需的常见 Module（如 [oneflow.nn.Conv2d](https://oneflow.readthedocs.io/en/master/nn.html?highlight=oneflow.nn.Conv2D#oneflow.nn.Conv2d)，[oneflow.nn.ReLU](https://oneflow.readthedocs.io/en/master/nn.html?highlight=oneflow.nn.ReLU#oneflow.nn.ReLU) 等等）。 用于搭建网络的所有 Module 类都继承自 [oneflow.nn.Module](https://oneflow.readthedocs.io/en/master/module.html#oneflow.nn.Module)，多个简单的 Module 可以组合在一起构成更复杂的 Module，用这种方式，用户可以轻松地搭建和管理复杂的神经网络。

```python
import oneflow as flow
import oneflow.nn as nn
```


## 定义 Module 类

`oneflow.nn` 下提供了常见并基础的 Module 类，我们可以在他们的基础上，通过自定义 Module 类搭建神经网络。搭建过程包括：

- 写一个继承自 `oneflow.nn.Module` 的类
- 实现类的 `__init__` 方法，在其中构建神经网络的结构
- 实现类的 `forward` 方法，这个方法针对 Module 的输入进行计算

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

以上代码，会输出刚刚搭建的 `NeuralNetwork` 网络的结构：

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

接着，调用 `net` （注意：推荐直接调用 `forward`）即可完成推理：

```python
X = flow.ones(1, 28, 28)
logits = net(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```

会得到类似以下的输出结果：

```text
Predicted class: tensor([1], dtype=oneflow.int32)
```

以上从数据输入、到网络计算，最终推理输出的流程，如下图所示：

![todo](imagetodo)

## `flow.nn.functional`

除了 `oneflow.nn` 外，[oneflow.nn.functional](https://oneflow.readthedocs.io/en/master/functional.html) 名称空间下也提供了不少 API。它与 `oneflow.nn` 在功能上有一定的重叠。比如 [nn.functional.relu](https://oneflow.readthedocs.io/en/master/functional.html?highlight=relu#oneflow.nn.functional.relu) 与 [nn.ReLU](https://oneflow.readthedocs.io/en/master/nn.html?highlight=relu#oneflow.nn.ReLU) 都可用于神经网络做 activation 操作。

两者的区别主要有：

- `nn` 下的 API 是类，需要先构造实例化对象，再调用；`nn.functional` 下的 API 是作为函数直接调用
- `nn` 下的类内部自己管理了网络参数；而 `nn.functional` 下的函数，需要我们自己定义参数，每次调用的时手动传入

实际上，OneFlow 提供的大部分 Module 是通过封装 `nn.functional` 下的方法得到的。`nn.functional` 提供了更加细粒度管理网络的可能。 

以下的例子，使用 `nn.functional` 中的方法，构建与上文中 `NeuralNetwork` 类等价的 Module `FunctionalNeuralNetwork`，读者可以体会两者的异同：

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
        out = flow.F.matmul(x, self.weight1)
        out = out + self.bias1
        out = nn.functional.relu(out)

        out = flow.F.matmul(out, self.weight2)
        out = out + self.bias2
        out = nn.functional.relu(out)

        out = flow.F.matmul(out, self.weight3)
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


## `Module Container`

除了上述模块，`Module `另外一个重要的概念是模型容器 (Containers)，常用的容器有 3 个，这些容器都是继承自`nn.Module`。 

-  nn.Sequetial：按照顺序包装多个网络层 

- nn.ModuleList： 像 python 的 list 一样包装多个网络层，可以进行迭代

- nn.ModuleDict：像 python 的 dict一样包装多个网络层，通过 (key, value) 的方式为每个网络层指定名称。 

### nn.Sequetial

 `nn.Sequetial`是`nn.Module`的容器，用于按顺序包装一组网络层，有以下两个特性。 

-  顺序性：各网络层之间严格按照顺序构建，我们在构建网络时，一定要注意前后网络层之间输入和输出数据之间的形状是否匹配 
-  自带`forward()`函数：在`nn.Sequetial`的`forward()`函数里通过 for 循环依次读取每个网络层，执行前向传播运算。这使得我们我们构建的模型更加简洁 

```python
import oneflow as flow
import oneflow.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
        
net = LeNet()
print("LeNet \n",net)       
```

我们也可以不使用容器构造神经网络，下面我们将对比使用容器和不使用容器构造简单的神经网络：
```python
import oneflow as flow
import oneflow.nn as nn

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = flow.nn.Linear(n_feature, n_hidden)
        self.predict = flow.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
net1 = Net(1, 10, 1)

class Net2(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net2, self).__init__()
        self.network=flow.nn.Sequential(
                flow.nn.Linear(n_feature, n_hidden),
                flow.nn.ReLU(),
                flow.nn.Linear(n_hidden, n_output)
            	)

net2 = Net2(1, 10, 1)
```

然后进行打印

```python
print(net1)
"""
Net (
  (hidden): Linear (1 -> 10)
  (predict): Linear (10 -> 1)
)
"""
print(net2)
"""
Sequential (
  (0): Linear (1 -> 10)
  (1): ReLU ()
  (2): Linear (10 -> 1)
)
"""
```

 我们可以发现，使用`flow.nn.Sequential`会自动加入`ReLU`, 但是 `net1` 中, `ReLU`实际上是在 `forward() `功能中才被调用的，所以使用容器会使得我们构建的模型更加简洁。




### nn.ModuleList

 `nn.ModuleList`是`nn.Module`的容器，用于包装一组网络层，以迭代的方式调用网络层，主要有以下 3 个方法： 

-  append()：在 ModuleList 后面添加网络层 

-   extend()：拼接两个 ModuleList 

-   insert()：在 ModuleList 的指定位置中插入网络层 

```python
import oneflow as flow
import oneflow.nn as nn

class ModuleList(nn.Module):
    def __init__(self):
        super(ModuleList, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(20)])

    def forward(self, x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
        return x

net = ModuleList()
print(net)
```



### nn.ModuleDict 

 `nn.ModuleDict`是`nn.Module`的容器，用于包装一组网络层，以索引的方式调用网络层，主要有以下 5 个方法： 

-  clear()：清空 ModuleDict 

-  items()：返回可迭代的键值对 (key, value) 

-  keys()：返回字典的所有 key 

-  values()：返回字典的所有 value 

-  pop()：返回一对键值，并从字典中删除 

下面的模型创建了两个`ModuleDict`：`self.choices`和`self.activations`，在前向传播时通过传入对应的 key 来执行对应的网络层。

```python
import oneflow as flow
import oneflow.nn as nn

class ModuleDict(nn.Module):
    def __init__(self):
        super(ModuleDict, self).__init__()
        self.choices = nn.ModuleDict({
            'conv': nn.Conv2d(10, 10, 3),
            'pool': nn.MaxPool2d(3)
        })

        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'prelu': nn.PReLU()
        })

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x


net = ModuleDict()
print(net)
```






## 神经网络实例AlexNet 

2012年，AlexNet横空出世，首次证明了学习到的特征可以超越手工设计的特征。AlexNet使用了8层卷积神经网络，以高出第二名 10 多个百分点的准确率赢得了2012年ImageNet图像识别挑战赛。它的出现一举打破了计算机视觉研究的现状，使得卷积神经网络开始在世界上流行，是划时代的贡献。

 AlexNet 特点如下： 

-  采用 ReLU 替换 Softmax ，缓解梯度消失  
-  采用 LRN (Local Response Normalization) 对数据进行局部归一化，减轻梯度消失 
-  采用 Dropout 提高网络的鲁棒性，增加泛化能力 
-  使用 Data Augmentation，包括 TenCrop 和一些色彩修改 

```python
import oneflow as flow
import oneflow.nn as nn

class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = flow.flatten(x, 1)
        x = self.classifier(x)
        return x
    
net=AlexNet()
print('AlexNet network',net)
```
