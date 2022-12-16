# Oneflow is compatible with PyTorch

With the alignment of OneFlow API and PyTorch, users can easily migrate the model from PyTorch to OneFlow. In this article, we will introduce three methods to port PyTorch code to OneFlow.

## 1. import oneflow as torch1

Change the original code ` import torch ` to:

```py
import oneflow as torch
```
https://gist.github.com/Artemisyuu/3185c4af8489ea72b4a09e7ad70b9acf

You can train the original model with OneFlow; However, you have to manually modify all files that contain `import torch` by this approach. In addition, it is necessary to modify the source code if a third-party library uses `torch`.

## 2. Using the command-line tool

Oneflow provides a command-line tool, which helps to simulate the environment of PyTorch within OneFlow’s Python Package and forward references of Pytorch to the real OneFlow module. The specific steps are as follows:


Enabling the simulation of PyTorch

```shell
eval $(oneflow-mock-torch)
```
https://gist.github.com/Artemisyuu/c6bfb433d6fd477a9b865c5cf76cd388

or

```shell
eval $(python3 -m oneflow.mock_torch)
```
https://gist.github.com/Artemisyuu/163541a7bb5d3a4474c14bfeafe6f580

After running the above command, you can observe the effect in the following example.

```py
import torch
print(torch.__file__)
import oneflow as flow
x = torch.zeros(2, 3)
print(isinstance(x, flow.Tensor))
```
https://gist.github.com/Artemisyuu/88abb38a96ba5adeaf0c05b725d18433

Disabling the simulation of PyTorch

```shell
eval $(oneflow-mock-torch disable)
```
https://gist.github.com/Artemisyuu/253ddf77f3a0c575309b21439e490555

or

```shell
eval $(python3 -m oneflow.mock_torch disable)
```
https://gist.github.com/Artemisyuu/6d35e06ac5c9a0ce70ea2d969871ea1c

## 3. Using built-in functions of OneFlow

The mock function with fine granularity provided by our company allows users to control whether to enable this function for a piece of code.

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
https://gist.github.com/Artemisyuu/fde3fd51e30e8871d66cbe44ab7d77d8

You can turn off the mock function like this when it’s needed to use the real torch module. 

```py
with mock.disable():
    import torch
    print(torch.__file__)
```
https://gist.github.com/Artemisyuu/dfc375e618e7ebd872029bd81104648c

`mock.enable` and `mock.disable` can act as functions. For example, you want to train a model with OneFlow, but it needs to be loaded by PyTorch. You can use it with the following code:

```py
mock.enable()
...
with mock.disable()
    module = torch.load_module(...)
# train the module with oneflow
```
https://gist.github.com/Artemisyuu/f7a8f946a84c42a3249d6e3d5555a639

A dictionary with the value of module has been saved separately in enable and disable mode. When enable or disable is turned on and off, the dictionary will replace `sys.modules` and the global variables that the current module belongs to. Therefore, users are required to `import` the module they need in each mode, and the code below will raise an error `name 'torch' is not defined`  in `with` statment of disable.

```py
with mock.enable():
    import torch
with mock.disable():
    torch.ones(2, 3)
```
https://gist.github.com/Artemisyuu/925e559f0f5d00aa469c997cc8b05073

## Conclusion

With the alignment of OneFlow API and PyTorch, users can easily migrate the PyTorch code to OneFlow. As mentioned above, three methods are available to train the PyTorch model with OneFlow. This is how users can experience the ultimate performance of OneFlow. 