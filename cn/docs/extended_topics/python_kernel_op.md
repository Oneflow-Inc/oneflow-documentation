# 使用 Python 扩展 Op

## 背景介绍
OneFlow 将各种对于数据的处理都抽象成了算子（operator），简称 op。 op 是作用在输入 tensor 上的操作，并将操作的结果写到输出 tensor 上。OneFlow 内部已经提供了比较完备的 op 算子，可以在 [ops 目录](https://github.com/Oneflow-Inc/oneflow/tree/master/oneflow/python/ops)下找到。

当 OneFlow 已有的 Python 算子及其组合无法满足构建神经网络的需求，或者 Python 层次的算子无法满足性能需求时，我们可以开发自定义 op。OneFlow 提供了两类开发自定义 Op 的途径，一类是以 Python 为主的 `Python Kernel` 开发，另外一类是[使用 C++ 扩展 Op](user_op.md)一文介绍的 `C++ Kernel` 开发。

`Python Kernel` 因为主要采用 Python 进行扩展，开发流程较简单，适用于快速预研、算法验证等场景。`C++ Kernel` 效率高，适用于开发已经验证稳健性并追求性能的算子。

本文将介绍介绍算子开发的背景知识和基本概念，并展示如何开发 `Python Kernel`。

### 基本概念
在进行 OneFlow 算子开发前，需要了解 `op_type_name`、`Op` 以及 `Kernel` 这几个概念：

- op_type_name：op_type_name 是 op 类别的全局唯一 ID， OneFlow 通过 op_type_name 查询并确认 op 的种类，进而实例化 op，用于构建计算图。op 的种类与 op 的关系，类似于类与对象的关系。
- op：逻辑上的算子，包含构图推理时的输入输出形状等信息，不包含具体的处理数据的逻辑。
- kernel：对于一个逻辑上的 op，在运行时，处理的逻辑会因为物理设备以及数据类型的不同。运行时的具体处理逻辑，由 kernel 完成。简单而言，op 与 kernel 是一对多的关系，我们可以使用 Python 完成具体运算，这样的Kernel 称为 `Python Kernel`，也可以[使用 C++ 开发 Kernel](./user_op.md)。
- OneFlow 的内核由 C++ 实现，但是用户接口使用 Python，因此需要按照约定编写 `Python Wrapper`，使得 Python Op 接口能与 C++ 内核交互。

### 开发步骤
使用 Python 扩展 Op，应该准备一个以 `op_type_name` 命名的目录，在该目录下，按照约定放置必需的文件，以 [oneflow/python/test/custom_ops/user_sigmoid](https://github.com/Oneflow-Inc/oneflow/tree/master/oneflow/python/test/custom_ops/user_sigmoid) 为例：

```text
user_sigmoid
├── user_sigmoid_cpp_def.cpp
├── user_sigmoid_py_api.py
└── user_sigmoid_py_kernel.py
```

其中：

- `op_type_name_cpp_def.cpp`(以上的 `user_sigmoid_cpp_def.cpp`) 文件中放置 Op 定义信息
- `op_type_name_py_api.py`(以上的 `user_sigmoid_py_api.py`)文件中放置 `Python Wrapper`，通过 `oneflow.user_op_builder` 将实现的 `Python Kernel` 导出给用户使用
- `op_type_name_py_kernel.py`(以上的 `user_sigmoid_py_kernel.py`)文件中放置 Python 实现的自定义算子的前向计算逻辑和后向计算逻辑

在下文中，我们将分别介绍：

- 如何编写 `op_type_name_cpp_def.cpp` 文件，定义 Op 信息
- 如何编写 `op_type_name_py_api.py` 文件，封装 Op 的 Python 接口
- 如何编写 `op_type_name_py_kernel.py` 文件，使用 Python 实现 Op 的计算 Kernel
- 在 OneFlow 中如何使用 `Python Kernel` 类型的自定义 Op

下文中，我们将介绍如何用 Python 实现一个自定义的 `user_relu` Op。

## Op 的实现与注册
首先，我们在 `user_relu_cpp_def.cpp` 中定义 op 并完成注册：
```cpp
#include "oneflow/core/framework/framework.h"

namespace oneflow {
namespace {

REGISTER_USER_OP("user_relu_forward")
  .Input("in")
  .Output("out")
  .SetTensorDescInferFn(
      [](user_op::InferContext *ctx) -> Maybe<void> {
        *ctx->Shape4ArgNameAndIndex("out", 0) =
            *ctx->Shape4ArgNameAndIndex("in", 0);
        *ctx->Dtype4ArgNameAndIndex("out", 0) =
            *ctx->Dtype4ArgNameAndIndex("in", 0);
        return Maybe<void>::Ok();
      });
}  // namespace
}  // namespace oneflow
```

分析以上代码：

- `oneflow/core/framework/framework.h` 中包含了我们创建一个 op 所需要的所有接口
- 与自定义 op 有关的接口集中在 `oneflow::user_op` 中，使用名称空间 `oneflow` 可以简化类型名称
- 宏 `REGISTER_USER_OP` 用于注册 op，其接受的参数 `user_relu_forward` 是 `op_type_name`。
- 使用 `REGISTER_USER_OP` 注册后，其实会返回一个 `OpRegistry` 类（位于[user_op_registry.h]()），通过调用该类方法，完成对自定义 op 的设置：
    1. `Input("in")` 表示其有一个名为 "in" 的输入
    2. `Output("out")` 表示其有一个名为 "out" 的输出
    3. `SetTensorDescInferFn` 用于设置形状及数据类型推导函数，描述该算子的输出的形状及类型与输入的关系。以上代码中，输出的形状、数据类型与输入的一致

`op_type_name_cpp_def.cpp` 文件是实现 `Python Kernel` 过程中唯一会使用到的 C++ 文件，它用于设置 Op 的信息，在现阶段，还无法将使用 C++ 配置 Op 的步骤省略（因为设置分布式等高级信息时必需），不过可以看到，该文件并不涉及具体的运算，仅仅是用于描述 Op，即使不熟悉 C++，根据我们的示例，也可以很轻松地掌握。

## 封装 Op 的 Python 接口
为了用户可以在 Python 层使用刚刚设置并注册的 `user_relu` Op，我们需要创建一个 `user_relu_py_api.py` 文件，其内容如下：

```python
import oneflow as flow

def user_relu_forward(x):
    op = (
        flow.user_op_builder("myrelu")
        .Op("user_relu_forward")
        .Input("in", [x])
        .Output("out")
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()
```

`flow.user_op_builder("op_myrelu")` 其实会返回一个名为 `op_myrelu` 的 `UserOpConfBuilder` 对象。

该对象包含 `Op`、`Input` 等方法，用于封装自定义 op，具体解释如下：

- `Op("user_relu_forward")`：参数必须为之前在 C++ 注册时的 `op_type_name`，OneFlow 通过它找到已经注册的 op 类型，并实例化 op 对象。
- `Input("in", [input_blob])`：对应了 C++ 中 op 注册时的 `Input`，第一个参数字符串必须与 C++ 注册 op 时的 `Input` 设置的字符串一致。第二个参数为输入的张量，是一个 `list`，因为一个 op 允许有多个输入。
- `Output("out")`：对应了 C++ 中 op 注册时的 `Output`。
- `Build`：以上设置完成后，调用 `Build` 可以得到自定义 op 的 Python wrapper

以下代码，将获取自定义 op 的输出：
```python
return op.InferAndTryRun().SoleOutputBlob()
```

其中的 `InferAndTryRun` 完成推导，返回 `UserOp`，如果返回结果只有一个输出，则使用 `SoleOutputBlob` 即可获取该唯一输出，否则，可以使用 `RemoteBlobList` 获取包含多个输出的列表。

## 使用 Python 实现 Kernel
如本文开始所描述，Op 只是逻辑上的概念，真正的计算需要 Kernel 完成，在 OneFlow 中可以既可以使用 C++ 也可以使用 Python 实现 Kernel，本文只介绍最易上手的 Python Kernel 的实现方法。使用 C++ 实现 Kernel 可以参考[使用 C++ 开发 Kernel](./user_op.md)。

为了为我们上文设置的 `user_relu` Op 提供 Python Kernel，我们需要创建一个 `user_relu_py_kernel.py` 文件，其内容如下：

```python
import numpy as np

def forward(args):
    (x,) = args
    y = (x>0)*x
    return y
```

以上的 `forward` 方法是必需实现的，它的实现对应了我们 Op 的 Python Kernel。关于它的约定有：

- 方法名必需为 `forward`
- 参数只有一个，类型为 `tuple`，`tuple` 中的元素个数和顺序，与 Op 注册时的 `Input` 对应。如我们之前为 `user_relu` 注册了 `Input("in")`，那么以上代码中 `(x, ) = args` 中的 `x` 就取到 `in` 的值
- 输出与 Op 注册时的 `Output` 对应
- 参数与返回值均为 `numpy` 对象，即不能（不会）是字符串、整型数字等其它类型

## 使用自定义 Op
完成以上工作后，我们得到了一个名为 `user_relu` 的目录，包含三个文件，它们的结构如下：

```text
user_relu/
├── user_relu_cpp_def.cpp
├── user_relu_py_api.py
└── user_relu_py_kernel.py
```

我们可以在 `user_relu` 文件夹所在的路径，创建一个测试文件，调用刚刚实现的自定义 Op，内容如下：

```python
import oneflow as flow
import numpy as np
import os
import oneflow.typing as tp

# 根据指定的路径与 op_type_name 创建 module 对象
module_path = os.path.dirname(os.path.abspath(__file__))
user_relu_op = flow.experimental.custom_op_module("user_relu", module_path)

# 使 Op, Python API, Python Kernel 生效
user_relu_op.py_api().cpp_def().py_kernel().build_load()

@flow.global_function()
def MyJob(x: tp.Numpy.Placeholder((5,), dtype=flow.float32)) -> tp.Numpy:
    return user_relu_op.api.user_relu_forward(x)

if __name__ == "__main__":
    input = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    output = MyJob(input)
    print(input)
    print(output)
```

以上代码中，先通过 `flow.experimental.custom_op_module` 创建 module 对象，它接收两个参数，第一个参数为 `op_type_name`， 第二个参数为 `user_relu` 文件夹所在的路径。返回的 `module` 对象，代表了我们自定义的 Op。

接着，通过 `user_sigmoid_op.py_api().cpp_def().py_kernel().build_load()` 可以使自定义 Op 生效，生效后的 Op 的 Python 接口，就是定义在 `user_relu_py_api.py` 文件中的方法名(`user_relu_forward`)，它被放置在 `moudle` 对象的 `api` 名称空间中。因此，我们需要通过以下方式调用:

```python
user_sigmoid_op.api.user_relu_forward(x)
```

## 为自定义 Op 提供反向计算
我们通过上述工作，已经完成了 `user_relu` 算子的正向计算过程，可以用于 `type="predict"` 的作业函数。但是，如果想支持 `type="train"` 类型的训练作业函数，我们就还需要为自定义 Op 提供反向计算。

为自定义 Op 提供反向计算的代码，需要写在 `op_type_name_cpp_def.cpp` 文件中，通过宏 `REGISTER_USER_OP_GRAD` 进行注册。

从数学角度上看，注册过程就是我们为自定义的 op，指定后向求梯度的计算方法。从编程角度看，就是为自定义 op 设置一个后向生成函数，在该函数中，编写代码，指定这个 op 的输入梯度的计算方法。

以下，我们将专门实现一个 Op，名为 `user_relu_backward`。我们将在为 `user_relu` 注册后向梯度时，用到这个“专门定制”的 Op。

### 实现 `user_relu_backward` Op
实现 `user_relu_backward` Op 的过程与实现 `user_relu` 的前向几乎是一样的。首先，在 `user_relu_cpp_def.cpp` 中设置并注册该 Op：

```cpp
REGISTER_USER_OP("user_relu_backward")
    .Input("y")
    .Input("dy")
    .Output("dx")
    .Attr<std::string>("device_sub_tag", "py")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* dy_shape = ctx->Shape4ArgNameAndIndex("dy", 0);
      Shape* dx_shape = ctx->Shape4ArgNameAndIndex("dx", 0);
      *dx_shape = *dy_shape;
      return Maybe<void>::Ok();
    });
```

值得注意的是，以上代码中 `.Attr<std::string>("device_sub_tag", "py")` 必不可少，因为这个 `user_relu_backward` 不是被用户直接调用，而是 OneFlow 框架自动求导中调用，我们需要设置 `device_sub_tag` 属性，告之 OneFlow 该 Op 的实现是 Python Kernel。

同理，因为不需要用户直接调用这个 `user_relu_backward` Op，因此我们不需要在 `user_relu_py_api.py` 为 `user_relu_backward` 封装 Python 接口。可以直接实现它的 Python Kernel。

在 `user_relu_py_kernel.py` 中，实现 `backward` 方法：

```python
def backward(args):
    (y, dy) = args
    dx = (y>0)*dy
    return dx
```
它的参数是一个 `tuple`，数目和顺序对应了 Op 注册时的 `Input`，输出对应了 Op 注册时的 Output。

### 为 Op 注册反向梯度
我们需要在 `user_relu_cpp_def.cpp` 中，通过宏 `REGISTER_USER_OP_GRAD` 为我们的正向 Op (`user_relu_forward`) 注册反向。

其代码如下：
```c++
REGISTER_USER_OP_GRAD("user_relu_forward")
    .SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) {
      const auto grad_op_name = ctx->FwOp().op_name() + "_grad";
      const auto& grad_op_func = [&ctx](user_op::BackwardOpBuilder& builder) {
        return builder.OpTypeName("user_relu_backward")
            .InputBind("y", ctx->FwOp().output("y", 0))
            .InputBind("dy", ctx->FwOp().output_grad("y", 0))
            .Output("dx")
            .Build();
      };
      ctx->DefineOp(grad_op_name, grad_op_func);

      const auto& dx_get_func = [&ctx, &grad_op_name]() -> const std::string& {
        return ctx->GetOp(grad_op_name).output("dx", 0);
      };
      ctx->FwOp().InputGradBind(user_op::OpArg("x", 0), dx_get_func);
    });
```

我们对以上代码进行解释，通过 `REGISTER_USER_OP_GRAD("user_relu_forward")` 注册为前向 Op 注册后向求梯度规则，该宏接收一个参数，就是 **前向的** `op_type_name`。

然后通过 `SetBackwardOpConfGenFn` 设置后向求梯度规则，同 Op 类似，在 `op_type_name_cpp_def.cpp` 中注册后向，其实不涉及真正的运算，而是设置后向计算与前向的对应关系，告诉 OneFlow 框架：

- 用什么 Op 求后向梯度
- 该 Op 的输入来自哪里，和前向 Op 什么关系

因此，以上代码中的：

```c++
      const auto& grad_op_func = [&ctx](user_op::BackwardOpBuilder& builder) {
        return builder.OpTypeName("user_relu_backward")
            .InputBind("y", ctx->FwOp().output("y", 0))
            .InputBind("dy", ctx->FwOp().output_grad("y", 0))
            .Output("dx")
            .Build();
      };
```

定义了 Op 求梯度的方法：使用 `user_relu_backward` 算子，并且将前向的输出 `y` 作为 `user_relu_backward` 的输入 `y`；将前向的输出 `y` 的梯度，作为 `user_relu_backward` 的输入 `dy`；最后输出 `dx`。

定完求梯度的方法后，需要调用
```cpp
ctx->DefineOp(grad_op_name, grad_op_func);
```
使之生效。

之后的代码：
```cpp
      const auto& dx_get_func = [&ctx, &grad_op_name]() -> const std::string& {
        return ctx->GetOp(grad_op_name).output("dx", 0);
      };
      ctx->FwOp().InputGradBind(user_op::OpArg("x", 0), dx_get_func);
```

是将前向的输入 `x` 和刚刚设置的求梯度的方法的输出(`dx`) 绑定到一起，这样，使用 OneFlow 训练时，就可以自动求导。

## 其它

- 本文涉及的代码可以在 [这里](https://github.com/Oneflow-Inc/oneflow-documentation/tree/master/cn/docs/code/extended_topics/python_op) 查看
- Op 注册的更多高级设置可以参考 [这里](user_op.md#opregistry)
- 注册反向梯度时，也可以使用已有的 Op，而无需专门定制反向 Op，可以参考 [这里](user_op.md#opgradregistry)
