## OneFlow—Eager动态图模式

除了追求极致性能的静态图模式外，OneFlow 还提供了 Eager 模式，即与 Pytorch 类似的动态图机制。Eager 模式使得模型网络的搭建以及 Debug 过程更加高效而迅速。熟悉 PyTorch 接口的用户，可以几乎没有成本地直接上手 OneFlow 的 Eager 模式。

在 [OneFlow v0.4.0](https://github.com/Oneflow-Inc/oneflow/releases/tag/v0.4.0) 版本开始，已经支持 Eager，可通过:

```python
import oneflow.experimental as flow
flow.enable_eager_execution()
```

轻松开启。在OneFlow的动态图模式下，包含

- flow.tensor,flow.Tensor 以及常见的 Tensor 初始化、矩阵运算等操作
- flow.nn.module以及常用的 nn.modules如：nn.Conv2d、nn.Linear、nn.CrossEntropyLoss等
- flow.nn.parameter
- flow.nn.optimizer以及常用的optimizers如 SGD、Adam等
- Autograd自动求导

基于上述模块已经可以轻松搭建常用网络，如：ResNet、BERT、MobileNetV3 等。



### Quick start

提供 LeNet 例子：

https://github.com/Oneflow-Inc/models/blob/main/quick_start_demo_lenet/lenet.py



### 新API文档

Eager 模式的 API 均附带了可以直接复制使用的代码例子：

https://oneflow.readthedocs.io/en/master/experimental.html



### Pytorch style ResNet50

https://github.com/Oneflow-Inc/models/tree/main/resnet50

后续版本将对齐/支持更多接口，届时可将大多数基于 PyTorch 搭建的网络，轻松切换到 OneFlow。

