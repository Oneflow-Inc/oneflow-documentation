# 搭建神经网络

​	 神经网络由对数据执行操作的层/模块组成。 `oneflow.nn` 命名空间提供了构建自己的神经网络所需的常见模块（例如`oneflow.nn.Conv2d`，`oneflow.nn.ReLU`等等）。 `oneflow` 中提供的每个模块都继承自 `nn.Module` ，神经网络可以由一个模块或者多个模块堆叠而成。这种嵌套的结构允许其轻松地构建和管理复杂的神经网络架构。   



## `flow.nn.Module` 与 `flow.nn.functional`

 	`oneflow`中`nn.Module`是面向对象的，可以保存状态。而`nn.functional` 是函数式的，无法保存状态。在大部分情况下，`oneflow`中的 `nn.module` 是通过封装 `nn.functional` 得到的。如果你需要更加细粒度使用一些API，推荐使用`nn.functional`提供的函数。 

`flow.nn.Module`和`flow.nn.functional`存在相同之处：

- 实际功能相同，例如 `flow.nn.Conv2d`和`flow.nn.functional.conv2d` 都是进行卷积 
-  运行效率近乎相同

两者之间还是存在一定差异：

-  **两者的调用方式不同** 。

   `nn.Module` 需要先进行实例化并传入参数，然后通过实例化对象和传入输入数据来调用它的`foward`函数完成计算。  

  ```python
  import oneflow as flow
  import oneflow.nn as nn
  import numpy as np
  
  inputs =flow.Tensor(np.random.randn(64, 3, 244, 244))
  conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
  outputs = conv(inputs)
  ```

   `nn.functional.xxx`  同时传入输入数据和weight, bias等其他参数 。 

  ```python
  import oneflow as flow
  import oneflow.nn as nn
  import numpy as np
  
  inputs = flow.Tensor(np.random.randn(33, 16, 30))
  weight = flow.Tensor(np.random.randn(64,3,3,3))
  bias = flow.Tensor(np.random.randn(64)) 
  outputs = nn.functional.conv2d(inputs, weight, bias, stride=[1], padding=[1], dilation=[1])
  ```

-  `nn.xxx`继承于`nn.Module`， 能够很好的与`nn.Sequential`结合使用， 而`nn.functional.xxx`无法与`nn.Sequential`结合使用。

  ```
  flow_layer = nn.Sequential(
              nn.Conv2d(3, 64, kernel_size=3, padding=1),
              nn.BatchNorm2d(num_features=64),
              nn.ReLU(),
              nn.MaxPool2d(kernel_size=2),
              nn.Dropout(0.2)
  )
  ```

- `nn.xxx`对存在weight参数的定义和管理是自动的 ；而`nn.functional.xxx`需要你自己定义weight，每次调用的时候都需要手动传入weight, 不利于代码复用。



## `nn.Module` 的常见方法及常见 `Module`

​	`nn.Module` 有 12 个属性，其中有8个是`OrderDict`(有序字典)。 我们在创建神经网络的时候， `__init__()`方法中会调用父类`nn.Module`的`__init__()`方法，创建这 8 个属性。 

```python
class Module(object):
    def __init__(self):
        self.training = True
        self._consistent = False
        self._non_persistent_buffers_set = set()
        self._is_full_backward_hook = None

        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._backward_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._forward_pre_hooks = OrderedDict()
        self._state_dict_hooks = OrderedDict()
        self._load_state_dict_pre_hooks = OrderedDict()
        self._modules = OrderedDict()
```

-  _parameters 属性：存储管理 nn.Parameter 类型的参数 

-  _modules 属性：存储管理 nn.Module 类型的参数 

-  _buffers 属性：存储管理缓冲属性

-  5 个 ***_hooks 属性：存储管理钩子函数



其中比较重要的是`parameters`和`modules`属性。 

`nn.Parameter` 主要作用是作为nn.Module中的可训练参数使用。

常见的`nn.Module`，包括`nn.Conv1d()`、`nn.Conv2d()` 、`nn.Conv3d()`和`nn.Linear()`等。

下面我们将对`nn.Conv2d()`进行简单介绍：

```python
import oneflow as flow
import oneflow.nn as nn
import numpy as np

# 样本数为3,channel：3, w:9, H:9
x = flow.Tensor(np.random.randn(3, 3, 9, 9))
print(x.shape)

# case1 in_channels=3, out_channels=6, kernel_size=3
conv1 = nn.Conv2d(3, 6, 3)
x1 = conv1(x)
print("[case1] in_channels=3, out_channels=6, kernel_size=3: ", x1.shape)

# [case2] stride_size=3
conv2 = nn.Conv2d(3, 6, 3, 3)
x2 = conv2(x)
print("[case2] stride_size=3: ", x2.shape)

# [case3] kernel_size=(3,2)
conv3 = nn.Conv2d(3, 6, (3, 2))
x3 = conv3( x)
print("[case3] kernel_size=(3,2): ", x3.shape)

# [case4] padding=3
conv4 = nn.Conv2d(3, 6, 3, 1, 2)
x4 = conv4(x)
print("[case4]padding=3: ", x4.shape)

# [case5] dilation=3
conv5 = nn.Conv2d(3, 6, 3, 1, 0, 3)
x5 = conv5(x)
print("[case5]dilation=3: ", x5.shape)

# [case6] groups=3
conv6 = nn.Conv2d(3, 6, 3, groups=3)
x6 = conv6(x)
print("[case6]groups=3: ", x6.shape)
```

可以得到以下输出：

```shell
flow.Size([3, 3, 9, 9])
[case1] in_channels=3, out_channels=6, kernel_size=3:  flow.Size([3, 6, 7, 7])
[case2] stride_size=3:  flow.Size([3, 6, 3, 3])
[case3] kernel_size=(3,2):  flow.Size([3, 6, 7, 8])
[case4] padding=3:  flow.Size([3, 6, 11, 11])
[case5] dilation=3:  flow.Size([3, 6, 3, 3])
[case6] groups=3:  flow.Size([3, 6, 7, 7])
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
