# Oneflow is compatible with PyTorch

With the alignment of OneFlow API and PyTorch, users can easily migrate the model from PyTorch to OneFlow. In this article, we will introduce three methods to port PyTorch code to OneFlow.

## 1. import oneflow as torch

Change the original code ` import torch ` to:

```py
import oneflow as torch
```

You can train the original model with OneFlow; however, you have to manually modify all the files that contain `import torch` by this approach. In addition, it is necessary to modify the source code if a third-party library uses `torch`.

## 2. Using the command-line tool

Oneflow provides a command-line tool, which helps to mock the environment of PyTorch within OneFlow’s Python Package and forward references of Pytorch to the real OneFlow module. The specific steps are as follows:


Enabling the mocking of PyTorch

```shell
eval $(oneflow-mock-torch)
```

or

```shell
eval $(python3 -m oneflow.mock_torch)
```

After running the above command, you can observe the effect in the following example.

```py
import torch
print(torch.__file__)
import oneflow as flow
x = torch.zeros(2, 3)
print(isinstance(x, flow.Tensor))
```

Disabling the docking of PyTorch

```shell
eval $(oneflow-mock-torch disable)
```

or

```shell
eval $(python3 -m oneflow.mock_torch disable)
```

## 3. Using built-in functions of OneFlow

We provide the mock function with fine granularity, and the users can determine whether to enable this function for a piece of code.

In the following `with` statement, the PyTorch module imported is OneFlow.

```py
import oneflow.mock_torch as mock
with mock.enable():
    import torch
    print(torch.__file__)
    import oneflow as flow
    x = torch.zeros(2, 3)
    print(isinstance(x, flow.Tensor))
```

You can turn off the mock function like this when it’s needed to use the real torch module. 

```py
with mock.disable():
    import torch
    print(torch.__file__)
```

`mock.enable` and `mock.disable` can act as functions. For example, if you want to train a model with OneFlow, but it needs to be loaded by PyTorch. Then, you can use it with the following code:

```py
mock.enable()
...
with mock.disable()
    module = torch.load_module(...)
# train the module with oneflow
```

A dictionary with the value of module is saved separately in enable and disable mode. When you turn enable or disable on and off, the dictionary will replace `sys.modules` and the global variables that the current module belongs to. Therefore, users are required to `import` the module they need in each mode, and the code below will raise an error `name 'torch' is not defined`  in `with` statment of disable.

```py
with mock.enable():
    import torch
with mock.disable():
    torch.ones(2, 3)
```

## Conclusion

With the alignment of OneFlow API and PyTorch, users can easily migrate the PyTorch code to OneFlow. As mentioned above, three methods are available to train the PyTorch model with OneFlow. This is how users can experience the ultimate performance of OneFlow. 
