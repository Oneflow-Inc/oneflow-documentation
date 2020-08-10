# 自定义 Op

## 背景介绍

### 自定义 op 是什么
当 OneFlow 已有的 Python 算子及其组合无法满足构建神经网络的需求，或者 Python 层次的算子无法满足性能需求时，我们可以使用 C++ 开发 OneFlow 自定义 op。

OneFlow 提供了一套机制，我们在这套机制下编写自定义 op 并将其注册到 OneFlow 中，就可以在 Python 中使用自定义 op。

下图展示了 OneFlow 中自定义 op 的注册机制：

![OneFlow UserOp Existing System](imgs/oneflow_system_userop.png)

可以看到，在 OneFlow 框架中，与自定义 op 注册有关的 Registry 有三种：

* `GradRegistry`：管理梯度注册，用于反向图中自动求梯度

* `OpRegistry`：管理 op 注册，用于生成前向图及构建 `Task Graph`

* `KernelRegistry`：管理 kernel 注册，用于运行时执行用户编写的 kernel 逻辑

在具体的编程过程中，我们其实是用 C++ 编写自定义 op，并生成动态链接库(so)文件。在 Python 中加载对应的 so 文件，就可以使用该 so 文件中的自定义 op。

### 基本概念

* op：逻辑上的算子，包含构图推理时的输入输出形状等信息，不包含具体的处理数据的逻辑

* kernel：对于一个逻辑上的 op，在运行时，处理的逻辑会因为物理设备以及数据类型的不同。运行时的具体处理逻辑，由 kernel 完成。简单而言，op 与 kernel 是一对多的关系，我们需要为 op 所支持的所有物理设备及数据类型实现和注册 kernel

* 注册：通过注册可以建立自定义 op 与 OneFlow 框架的联系。在 OneFlow 中提供了一系列名如 `REGISTER_XXX` 的宏帮助完成 op、kernel 等的注册

* 加载动态库：自定义的 op 及其 kernel 等被链接为 动态库 so 文件，在 Python 中使用前需要先加载。 OneFlow 提供了 `oneflow.config.load_library` 接口加载自定义 op 的动态库文件

* Python wrapper：在 Python 中调用 C++ 层实现的自定义 op，需要在 Python 层编写一个 wrapper，OneFlow 提供了 `oneflow.user_op_builder` 接口完成该工作。


### 编写自定义 op 的步骤
1. 实现 op 并注册：op 的实现主要用于前向图构图，包括指定 op 的名称、输入、输出、配置属性以及一些必要的用于推导 tensor 的形状与数据类型的函数

2. 实现 op 对应的 kernel 并注册：kernel 负责运行时的具体运算过程，一个 op 可能会对应多个 kernel

3. （可选）实现 op 对应的 grad 并注册：如果自定义 op 需要支持后向展开，需要实现一个后向函数并注册

4. 编译链接得到 so 文件

5. 在 Python 中加载 so 文件，并且使用 `oneflow.user_op_builder` 封装 C++ 编写的自定义 op

6. 测试

## 示例
我们将实现一个支持 cpu 及 GPU 运算的 "myrelu" 自定义 op。
完整的代码见 [code/extended_topics/create_user_op]()。

### op 的实现与注册
我们在 `myrelu_op.cpp` 中定义了 op 并完成了注册：
```cpp
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

REGISTER_USER_OP("myrelu")
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
} // namespace

} // namespace oneflow
```

分析以上代码：

* `oneflow/core/framework/framework.h` 中包含了我们创建一个 op 所需要的所有接口

* 与自定义 op 有关的接口集中在 `oneflow::user_op` 中，使用名称空间 `oneflow` 可以简化类型名称

* 宏 `REGISTER_USER_OP` 用于注册 op，其接受的参数（`myrelu`）其实是 OneFlow 中用于查询 op 的 ID，必须全局唯一

* 使用 `REGISTER_USER_OP` 注册后，其实会返回一个 `OpRegistry` 类（位于 `oneflow\core\framework\user_op_registry.h`），通过调用该类方法，完成对自定义 op 的设置：
    1. `Input("in")` 表示其有一个名为 "in" 的输入
    2. `Output("out")` 表示其有一个名为 "out" 的输出
    3. `SetTensorDescInferFn` 用于设置形状及数据类型推导函数，描述该算子的输出的形状及类型与输入的关系。以上代码中，输出的形状、数据类型与输入的一致

### cpu kernel 的实现与注册
我们在 `myrelu_cpu_kernel.cpp` 中实现了 CPU 版本的 kernel 并注册：
```cpp
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

template <typename T>
void MyRelu(DeviceCtx *ctx, const int64_t n, const T *x, T *y) {
  T zero = (T)(0);
  for (int64_t i = 0; i != n; ++i) {
    y[i] = std::max(x[i], zero);
  }
}

template <DeviceType device_type, typename T>
class ReluKernel final : public user_op::OpKernel {
public:
  ReluKernel() = default;
  ~ReluKernel() = default;

private:
  void Compute(user_op::KernelComputeContext *ctx) const override {
    const user_op::Tensor *in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor *out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    MyRelu<T>(ctx->device_ctx(),
           in_tensor->shape().elem_cnt(),
           in_tensor->dptr<T>(), 
           out_tensor->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_RELU_KERNEL(device, dtype)          \
  REGISTER_USER_KERNEL("myrelu")                     \
      .SetCreateFn<ReluKernel<device, dtype>>()      \
      .SetIsMatchedHob(                              \
          (user_op::HobDeviceType() == device) &     \
          (user_op::HobDataType("out", 0)            \
            == GetDataType<dtype>::value));

REGISTER_RELU_KERNEL(DeviceType::kCPU, float)
REGISTER_RELU_KERNEL(DeviceType::kCPU, double)
} // namespace

} // namespace oneflow
```
在 OneFlow 中实现 kernel， 必须定义一个继承自 `oneflow::user_op::OpKernel` 的类，并重写其中的虚函数。

在以上代码中，重写了 `Compute` 与 `AlwaysComputeWhenAllOutputsEmpty` 两个虚函数，他们分别的意义是：

* `Compute` 必须重写，在其中实现具体的运算逻辑

* `AlwaysComputeWhenAllOutputsEmpty` 必须重写，对于绝大多数 op 而言直接返回 `false` 即可。对于极少数内部需要维护状态，因此即使输出为空也需要调用 kernel 进行计算的 op 而言，应该返回 `true`

实现 kernel 类后，需要调用 `REGISTER_USER_KERNEL` 注册。`REGISTER_USER_KERNEL("myrelu")` 所接受的字符串参数，其实是一个唯一的全局 ID，OneFlow 依据它完成注册和运行时查询工作，在 Python 层封装 op 时也需要使用这个 ID。

`REGISTER_USER_KERNEL("myrelu")` 会返回一个 `OpKernelRegistry` 对象，需要调用它的各个方法，设置注册信息。上文代码中涉及到

* `SetCreateFn<T>()`：该模板方法的模板参数 `T`，就是我们实现的 kernel 类，OneFlow 将使用它创建 kernel 对象。

* `SetIsMatchedHob`：因为一个 op 可能有多个 kernel，要想根据物理设备及数据格式的不同而选择不同的 kernel 进行计算，就需要调用 `SetIsMatchedHob` 进行设置。该方法接受一个表达式，表达式为 `true` 时，OneFlow 将调用该 kernel 完成计算。

### GPU kernel 的实现与注册
我们在 `myrelu_gpu_kernel.cpp` 中实现了 GPU 版本的 kernel 并注册：
```cpp
#include "oneflow/core/framework/framework.h"
#include <cub/cub.cuh>

namespace oneflow {
namespace {
template <typename T>
__global__ void ReluForwardGpu(const int n, const T *x, T *y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = x[i] > 0 ? x[i] : 0; }
}

class ReluGpuFloatKernel final : public user_op::OpKernel {
public:
  ReluGpuFloatKernel() = default;
  ~ReluGpuFloatKernel() = default;

private:
  void Compute(user_op::KernelComputeContext *ctx) const override {
    const user_op::Tensor *in_tensor = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor *out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);

    int32_t n = in_tensor->shape().elem_cnt();
    const float *in_ptr = in_tensor->dptr<float>();
    float *out_ptr = out_tensor->mut_dptr<float>();
    ReluForwardGpu<float>
        <<<32, 1024, 0, ctx->device_ctx()->cuda_stream()>>>(n, in_ptr, out_ptr);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_RELU_KERNEL(device, dtype)          \
  REGISTER_USER_KERNEL("myrelu")                     \
      .SetCreateFn<ReluGpuFloatKernel>()             \
      .SetIsMatchedHob(                              \
          (user_op::HobDeviceType() == device) &     \
          (user_op::HobDataType("out", 0)            \
            == GetDataType<dtype>::value));

REGISTER_RELU_KERNEL(DeviceType::kGPU, float)
REGISTER_RELU_KERNEL(DeviceType::kGPU, double)
} // namespace
} // namespace oneflow
```

可以看到， 实现并注册 GPU kernel 的过程与 CPU kernel 几乎一致。区别主要在于：

* 因为使用了 CUDA 编程，所以包含了 CUDA 对应的头文件

* `Compute` 内部使用了 GPU 的方法

* `SetIsMatchedHob` 中所匹配的设备为 GPU

此外，我们马上会在下文看到，因为使用了 CUDA，我们需要使用 nvcc 编译器（而不是 g++）来编译 GPU kernel。

### 编译链接选项说明
在 `oneflow.sysconfig` 下包含了 `get_compile_flags`、`get_include`、`get_lib`、`get_link_flags` 方法分别对应自定义 op 时的：

- 编译选项
- 头文件路径
- 链接库路径
- 链接选项

比如：
```text
>>> import oneflow
>>> oneflow.sysconfig.get_compile_flags()
['-I/home/yaochi/oneflow/build/python_scripts/oneflow/include', '-DHALF_ENABLE_CPP11_USER_LITERALS=0', '-DWITH_CUDA', '-D_GLIBCXX_USE_CXX11_ABI=0']
```

也可以通过命令行直接获取编译、链接选项：
```shell
python -c "import oneflow; print(' '.join(oneflow.sysconfig.get_compile_flags()))"
python -c "import oneflow; print(' '.join(oneflow.sysconfig.get_link_flags()))"
```

对于 GPU kernel，链接时还需要指定 `cudart` 库。

### 编译、链接得到动态库
对于这个简单示例，可以使用以下 Makefile 进行构建：
```bash
CFLAGS = $(shell python -c "import oneflow; print(' '.join(oneflow.sysconfig.get_compile_flags()))")
LFLAGS = $(shell python -c "import oneflow; print(' '.join(oneflow.sysconfig.get_link_flags()))")
CUDAPATH = /usr/local/cuda-10.1/lib64

all: final_relu.so

myrelu_op.o: myrelu_op.cpp
	g++ -std=c++11 -c myrelu_op.cpp \
	-o myrelu_op.o                  \
	-fPIC                           \
	${CFLAGS}                       \
	${LFLAGS}                       \
	-O2

myrelu_cpu_kernel.o: myrelu_cpu_kernel.cpp
	g++ -std=c++11 -c myrelu_cpu_kernel.cpp \
	-o myrelu_cpu_kernel.o                  \
	$(CFLAGS) -fPIC

myrelu_gpu_kernel.o: myrelu_gpu_kernel.cu 
	nvcc -std=c++11 -c myrelu_gpu_kernel.cu \
	-o myrelu_gpu_kernel.o                  \
	$(CFLAGS) -x cu -Xcompiler -fPIC

final_relu.so: myrelu_op.o myrelu_cpu_kernel.o myrelu_gpu_kernel.o
	g++ -std=c++11 myrelu_op.o \
	myrelu_cpu_kernel.o        \
	myrelu_gpu_kernel.o        \
	-shared -o final_relu.so   \
	$(CFLAGS)                  \
	-fPIC                      \
	-L$(CUDAPATH)              \
	-lcudart                   \
	$(LFLAGS)

clean:
	rm -rf *.so *.o
```

我们使用 `g++` 编译 `myrelu_op.cpp`、`myrelu_cpu_kernel.cpp`，使用 `nvcc` 编译 `myrelu_gpu_kernel.cpp`，最后得到的目标文件链接为 `final_relu.so`。

我们将在 Python 中加载 `final_relu.so` 并使用封装、使用自定义 op。

### 在 Python 使用自定义 op 
在 Python 中使用自定义 op 包括以下几个基本步骤：

* 使用 `oneflow.config.load_library` 加载 so 文件

* 使用 `oneflow.user_op_builder` 生成自定义 op 的 Python wrapper

* 调用以上的 Python wrapper 得到结果

以下代码在 Python 层次封装了 `myrelu` 并调用：
```python
import oneflow as flow
import numpy as np
import oneflow.typing as tp

# 加载模块
flow.config.load_library("final_relu.so")

# 默认配置
flow.config.gpu_device_num(1)

# python op wrapper function
def myrelu(input_blob):
    op = (
        flow.user_op_builder("op_myrelu")
        .Op("myrelu")
        .Input("in", [input_blob])
        .Output("out")
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


# 网络代码
@flow.global_function()
def MyJob(x: tp.Numpy.Placeholder((5,), dtype=flow.float32)) -> tp.Numpy:
    return myrelu(x)

if __name__ == "__main__":
    input = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    output = MyJob(input)
    print(input)
    print(output)
```
预期结果为：
```text
[-2. -1.  0.  1.  2.]
[0. 0. 0. 1. 2.]
```

以上代码中的：`flow.config.load_library("final_relu.so")` 为加载 so 文件。

我们重点介绍 `myrelu` 内部构建 python wrapper 并运行的过程。

`flow.user_op_builder("op_myrelu")` 其实会返回一个名为 `op_myrelu` 的 `UserOpConfBuilder` 对象，该对象包含 `Op`、`Input` 等方法，用于封装自定义 op，具体解释如下：

* `Op("myrelu")`：参数必须为 cpp 注册 op 时的字符串，OneFlow 通过该字符串建立 Python 层与 C++ 层的联系

* `Input("in", [input_blob])`：对应了 op 注册时的 `Input`，第一个参数字符串必须与 C++ 注册 op 时的 `Input` 设置的字符串一致。第二个参数为输入的 blob。

* `Output("out")`

* `Build`



## OpRegistry

## OpKernelRegistry

## UserOpConfBuilder

## 高级特性
到现在为止，我们已经完成`Relu` op的构建。当然，Relu Op是一个比较简单的Op，如果我们需要构建一个比较复杂的Op，就需要使用一些额外的高级特性来协助我们。

### Op Registration

#### Attribute

有的Op需要有配置属性，例如`Conv` op需要配置其`padding`的方式、`Reshape` op需要配置一个`tensor shape`。当Op被添加到Graph中时，我们就需要给这些属性设置合理的值了。

在OneFlow中，你可以在注册Op时指明其需要的属性Attr，此处我们以`Reshape`  op为例：

```cpp
REGISTER_USER_OP("Reshape")
    .Input("in")
    .Output("out")
    .Attr("shape", UserOpAttrType::kAtShape)
    .SetShapeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      Shape conf_shape = ctx->GetAttr<Shape>("shape");
      CHECK_EQ(in_shape->NumAxes(), conf_shape.NumAxes());
      *out_shape = conf_shape;
      return Maybe<void>::Ok();
    });
```

上述代码中，我们为`Reshape` op配置了一个attribute，名字为`shape`，其类型是`UserOpAttrType::kAtShape`。

在OneFlow中，我们目前支持了如下几种AttrType：

| UserOpAttrType | 对应的C++数据类型    |
| -------------- | -------------------- |
| kAtInt32       | int32_t              |
| kAtInt64       | int64_t              |
| kAtBool        | bool                 |
| kAtFloat       | float                |
| kAtDouble      | double               |
| kAtShape       | oneflow::Shape       |
| kAtListInt32   | std::vector<int32_t> |
| kAtListInt64   | std::vector<int64_t> |
| kAtListFloat   | std::vector< float > |

在`Reshape` op注册的`ShapeInferFn`中，其就把`shape`对应的Attr的值（`oneflow::Shape`类型）赋给了`out` tensor对应的`out_shape`。



除了指定Attr的类型，我们还可以为其配置一个默认值，默认值的类型即表格中对应的C++数据类型，如：

``` cpp
.Attr("is_transpose", UserOpAttrType::kAtBool, false)
    
.Attr("size", UserOpAttrType::kAtInt32, 10)
    
.Attr("vector_of_size", UserOpAttrType::kAtListInt32, std::vector<int32_t>{10, 11, 12})
```



#### Check Attribute Function

对于某些Attribute来说，其需要更详细的划定取值范围，这时就需要在注册Op时通过`CheckAttrFn`来指定其取值范围。

例如，对于`Conv` op来说，其有一个配置选项`data_format`，其类型是string字符串，但取值只能是`channels_first`或`channels_last`这两个，除此之外都不合法，所以我们需要指定其范围，注册Op时就需要如下指定：

```cpp
.Attr("data_format", UserOpAttrType::kAtString, std::string("NCHW"))
.SetCheckAttrFn([](const user_op::UserOpDefWrapper& def,
                   const user_op::UserOpConfWrapper& conf) -> Maybe<void> {
   std::string data_format = conf.attr<std::string>("data_format");
   if (data_format == "channels_first" || data_format == "channels_last") { return Maybe<void>::Ok(); }
   return oneflow::Error::CheckFailed()
         << "data_format value: " << data_format << " for Conv op is illegal.";
})
```



#### Multiple In/Out

对于有些Op来说，其共享同一个name的 in/out tensor可能有多个，例如对于`Add` op来说，其`input`这个 name下可能对应有多个tensor，这时我们就需要在注册Op时指定其对应的输入输出的个数。

OneFlow框架支持对Op的Input/Output做如下配置：

```cpp
.Input("input") // input 必须对应有1个tensor

.Input("input", 5) // input 必须对应有5个tensor
    
.InputWithMinimum("input", 5) // input 必须对应至少5个tensor
    
.OptionalInput("input") // input 可能没有对应的tensor，若有则须对应1个tensor
    
.OptionalInput("input", 5) // input 可能没有对应的tensor，若有则须对应5个tensor
    
.OptionalInputWithMininum("input", 5) // input 可能没有对应的tensor，若有则须对应至少5个tensor
    
    
// Output与Input用法相同
```



#### DataType Infer Function

多数Op的input tensor和output tensor的类型相同，但对于一些特殊Op（如`Cast` op）来说，其需要传入一个data_type infer function来推导output tensor的类型。

``` cpp
.SetDataTypeInferFn([](user_op::InferContext* ctx) {
      DataType* out_tensor_type = ctx->Dtype4ArgNameAndIndex("out", 0);
      *out_tensor_type = DataType::kDouble;
      return Maybe<void>::Ok();
    })
```

上述代码就是把out tensor的数据类型设置为`double`。

对于无需更改数据类型的Op，也无需在注册Op时指定`SetDataTypeInferFn()`，因为OneFlow框架提供的默认实现就是让output tensors 和 input tensors 的数据类型一致。

### Kernel Registration

#### Temporary Buffer Size Infer Function

对于一些Op来说，其某种实现（即Kernel）可能会需要一些额外的buffer来存储一些临时数据。

在OneFlow框架中，这些临时的buffer也是作为tensor来在`Compute`函数中使用的。

而需要多大的临时buffer，就得我们在注册Kernel时指定了。

``` cpp
REGISTER_USER_KERNEL("XOp")
    .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) { return new XKernel(ctx); })
    .SetIsMatchedPred(...)
    .SetInferTmpSizeFn([](const oneflow::user_op::InferContext*) { return 1024; });
// XOp 对应的 XKernel 需要1024Byte大小的buffer

class XKernel final : public oneflow::user_op::OpKernel {
...
  void Compute(oneflow::user_op::KernelContext* ctx) override {
    ...
    oneflow::user_op::Tensor* tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    // 访问1024Byte的tensor
    ...
  }
};
```





### 注册Op的Grad 

- 说明及示例

Oneflow使用自动求导的方式进行后向计算图展开，为了对自定义的op进行后向求导，需要你注册一个后向生成函数来根据这个op的输出blob的导数计算输入blob的导数。你可以通过已有的其他op来构建这个后向展开的子图，当无法用已有op来描述后向时，你需要自己实现一个后向grad_op来表示。
后向生成函数在c++端注册。对于relu op，其后向生成函数的示例如下：
```cpp
#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP_GRAD("Relu").SetBackwardOpConfGenFn([](user_op::BackwardOpConfContext* ctx) {
  const auto relu_grad_op_name = ctx->FwOp().op_name() + "_grad";
  ctx->DefineOp(relu_grad_op_name, [&ctx](user_op::BackwardOpBuilder& builder) {
    return builder.OpTypeName("relu_grad")
        .InputBind("y", ctx->FwOp().output("out", 0))
        .InputBind("dy", ctx->FwOp().output_grad("out", 0))
        .Output("dx")
        .Build();
  });
  ctx->FwOp().InputGradBind(user_op::OpArg("in", 0), [&ctx, &relu_grad_op_name]() {
    return ctx->GetOp(relu_grad_op_name).output("dx", 0);
  });
});

}  // namespace oneflow
```
- 后向生成函数的步骤：

宏`REGISTER_USER_OP_GRAD(op_type_name).SetBackwardOpConfGenFn(fn);`用来注册你的自定义op的后向生成函数，其中fn函数带有一个`BackwardOpConfContext* ctx`参数，带有生成Op需要的信息。

在生成后向图的函数中，图里的blob是由一个叫logical blob name的字符串表示的，其中包含了产生这个blob的op的name，以及这个blob在这个op里的name。后向生成函数的任务是对前向op的输入blob，生成一个op的子图，这个子图接收前向op的输入输出blob以及输出blob的导数（梯度/grad）blob，子图的最终输出是前向op输入blob对应的导数（梯度）blob，因此针对每个（可能）需要生成梯度的blob，都需要构建一个由其他op组成的子图，并将子图的输出blob与这个需要生成梯度的blob绑定。编写这个生成子图的过程通常包含下面几步：
  1. 使用`ctx->DefineOp()`和`BackwardOpBuilder`来构建这个子图中的new_op，通常这些new_op的输入是前向op的in/out或者out对应的out_grad；
  2. 使用`ctx->FwOp().InputGradBind()`和`ctx->GetOp()`将前向op的输入blob绑定到子图的输出blob的logical blob name上；

- 可能用到的接口介绍：
  1. `ctx->FwOp()`：获取前向Op
    * `.InputGradBind(input_arg, grad_get_fn)` 会自动判断前向op的输入是否需要生成后向的梯度，如果需要会触发`grad_get_fn`的执行，进行前向输入和后向梯度的绑定，其中`grad_get_fn`中都会调用`ctx->GetOp()`来触发之前定义的op的创建并获取结果；
    * `.input(arg_name,index)` 得到输入的logical blob name
    * `.output(arg_name,index)` 得到输出的logical blob name
    * `.output_grad(output_arg_name, index)` 返回前向op的输出对应的后向梯度blob的logical blob name
    * `.attr(attr_name)` 得到op的属性值
    * `.arg_tensor_desc(arg_name, index)` 返回前向op的输入/输出对应的TensorDesc，包含shape、dtype信息
  2. `ctx->DefineOp(op_name, build_fn)`:定义名为`op_name`的Op的创建函数`build_fn`
    * 当调用`ctx->GetOp(op_name)`会触发`build_fn`进行Op创建，如果Op已经被创建过，那么这里直接获取创建的结果；
  3. `BackwardOpBuilder`: 创建Op的类
    * `.OpTypeName(op_type_name)`  指定这个op的type
    * `.InputBind(arg_name, logical_blob_name)`  可选项，可以调用多次，每次指定一个input_arg_name，同时传入一个logical_blob_name，表明这个input arg name对应的blob。如果该input_arg_name对应多个输入blob，则调用`.Input()`的顺序就是其对应的index顺序
    * `.Output(arg_name, num)`  可选项，可调用多次，每次指定一个`output_arg_name`实际对应的输出blob的数量，也可以调用 `.Output(arg_name)`，表示`num = 1`
    * `.Attr(attr_name, val)` 可选项，可调用多次，每次指定一个attr属性的属性名称和参数值，表示对这个attr赋值为val
    * `.Build()`  返回结果，表示你构建完毕的新op
  4. `ctx->GetOp(op_name)`: 得到`op_name`对应Op创建好后返回的结果，Op只有被`ctx->GetOp`获取时才会被真正创建，这里实现了Op子图的惰性创建过程
    * `.input(arg_name,index)` 得到输入的logical blob name
    * `.output(arg_name,index)` 得到输出的logical blob name
    * `.attr(attr_name)` 得到op的属性值