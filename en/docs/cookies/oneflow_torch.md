# OneFlow and PyTorch Compatibility
The APIs of OneFlow align with PyTorch's, users can move their PyTorch models to OneFlow easily. This article describes three ways to migrate your PyTorch code to OneFlow.
## 1. import oneflow as torch
Change your `import torch` to
```py
import oneflow as torch
```
and use OneFlow to train the original models.
## 2. Use Command Line Tool by OneFlow
OneFlow provides a command line tool, mocking PyTorch environment in OneFlow's Python Package, and forward any imports from this module to the actual OneFlow module. 

Turn on mocking PyTorch
```
eval $(python3 -m oneflow.mock_torch)
```
or
```
eval $(oneflow-mock-torch)
```
where the default choice of argument `mock` is `enable`.

Turn off mocking PyTorch
```
eval $(python3 -m oneflow.mock_torch disable)
```
or
```
eval $(oneflow-mock-torch disable)
```
## 3. Use OneFlow Function
work in progress.
