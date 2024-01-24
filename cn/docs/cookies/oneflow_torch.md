# OneFlow 模拟 PyTorch

OneFlow 的 API 与 PyTorch 对齐，用户可以很方便地将 PyTorch 模型迁移到 OneFlow。本文介绍三种方法，将使用 PyTorch 运行的代码迁移到 OneFlow。

## 1. import oneflow as torch

将原有的`import torch`的代码改为

```py
import oneflow as torch
```

就可以使用 OneFlow 训练原有模型；然而，这种方法需要手动修改所有 `import torch` 的文件，如果第三方库使用了 `torch`，还需要更改第三方库的源码。

## 2. 使用命令行工具

OneFlow 提供了一个命令行工具，在 OneFlow 的 Python Package 内模拟了 PyTorch 环境，并将对该模块的引用都转发到实际的 OneFlow 模块中。具体的用法如下

开启模拟 PyTorch

```shell
eval $(oneflow-mock-torch)
```

或

```shell
eval $(python3 -m oneflow.mock_torch)
```

为了便于调试，OneFlow 为该方法提供了两个参数：

1.  lazy 参数，`lazy=True` 时，对不存在的接口会返回一个假对象而不立即报错。**建议将该参数设置为 True**，这样即便您 import 的第三方库中含有 OneFlow 暂时不存在的接口，只要没有实际使用到该接口，mock torch 也能正常工作。

2.  verbose 参数，如果同时设置 `verbose=True`，会打印出有哪些假对象被访问或使用，便于调试。
用法如下

开启模拟 PyTorch，并配置 lazy 和 verbose 参数

```shell
eval $(oneflow-mock-torch --lazy --verbose)
```

或

```shell
eval $(python3 -m oneflow.mock_torch --lazy --verbose)
```


运行上述命令后，通过以下示例观察效果

```py
import torch
print(torch.__file__)
import oneflow as flow
x = torch.zeros(2, 3)
print(isinstance(x, flow.Tensor))
```

关闭模拟 PyTorch

```shell
eval $(oneflow-mock-torch disable)
```

或

```shell
eval $(python3 -m oneflow.mock_torch disable)
```

## 3. 使用 OneFlow 的内置函数

我们提供了更细粒度的 mock 功能，用户可以自行控制某段代码是否启用 mock 功能。
如下的 `with` 语句中，导入的 PyTorch 模块实际上是 OneFlow

```py
import oneflow.mock_torch as mock
with mock.enable():
    import torch
    print(torch.__file__)
    import oneflow as flow
    x = torch.zeros(2, 3)
    print(isinstance(x, flow.Tensor))
```

同样 OneFlow 为 `mock.enable()` 提供了便于调试的参数 lazy 和 verbose，可以这样设置 

`with mock.enable(lazy=True, verbose=True)`


当你需要使用真正的 torch 模块时，可以这样关闭 mock 功能

```py
with mock.disable():
    import torch
    print(torch.__file__)
```

`mock.enable` 和 `mock.disable` 也可以作为函数使用，例如，对于一段用户想要用 OneFlow 进行训练的模型，而该模型需要 PyTorch 来加载，可以这样使用

```py
mock.enable()
...
with mock.disable()
    module = torch.load_module(...)
# train the module with oneflow
```

enable 模式和 disable 模式各自保存了一份值为模块的字典，在开关enable/disable时会替换 `sys.modules` 和当前所属模块的全局变量，故用户需要在 enable 模式和 disable 模式时自行 `import` 需要的模块，如下代码会在 disable 的 `with` 语句里报 `name 'torch' is not defined` 的错
```py
with mock.enable():
    import torch
with mock.disable():
    torch.ones(2, 3)
```

## 总结

由于 OneFlow 的 API 与 PyTorch 对齐，用户能够将 PyTorch 代码很方便地迁移到 OneFlow。以上介绍了三种使用 OneFlow 来训练 PyTorch 模型的方法，希望用户能够体验到 OneFlow 极致的性能。
