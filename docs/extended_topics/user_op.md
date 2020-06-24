# 创建新的Op

## 背景
本文档服务于那些需要自定义实现C++ Op的用户，通常来说，这来源于如下需求

- 自定义Op难以通过OneFlow已有的Op在Python前端组合搭配生成。
- 自定义Op可以通过OneFlow已有Op在Python前端组合搭配生成，但却满足不了性能需求。
- 用户想要手动对kernel进行融合fuse，以满足某些特定的需求。

除上述需求之外的需求，我们推荐您在Python前端使用现有Op组合得到想要的Op。



对于的确需要手动实现C++ Op的用户，你需要如下步骤来让自定义Op正常工作：

1. 注册Op的定义： Op的定义独立于Op的实现（即Kernel），用来描述Op的功能性。通常包括Op的名称，Op的输入和输出，Op的配置属性和一些必要的用于推导Tensor的shape和data type的函数。
2. 用C++实现Op对应的Kernel： Kernel用来描述Op的详细计算过程。对于一个Op来说，可能会对应多个Kernel。
3. 书写对应的Python前端：因为OneFlow的网络构建是基于Python去书写的，所以我们需要在Python前端去书写少量代码来封装前两步所写的C++的代码。
4. （可选）注册Op对应的后向构图函数：如果训练网络中需要Op的反向，那么我们还需要去写一个函数来告诉OneFlow如何在去构建该Op对应的后向计算过程。
5. 测试Op：上述步骤全部执行完毕后，我们还需要对Op进行测试以保证Op的正确性，具体步骤见后续。


## 定义新的Op

我们以`Relu`为例，来走一遍Op注册的流程，构建一个cpp文件`relu.cpp`用于注册Op。

``` cpp
#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP("Relu")
    .Input("in")
    .Output("out")
    .SetShapeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      Shape* in_shape = ctx->Shape4ArgNameAndIndex("in", 0);
      Shape* out_shape = ctx->Shape4ArgNameAndIndex("out", 0);
      *out_shape = *in_shape;
      return Maybe<void>::Ok();
    });   
    
}
```

分析下上述代码：

首先是include了`oneflow/core/framework/framework.h`。`framework.h`中包括了我们创建一个Op需要的所有头文件，所以我们只需要include这一个头文件就行。

其次，为了简化类型名，我们在`namespace oneflow`下书写注册Op定义的相关代码。

最后，我们来看下注册的代码。`Relu`  op只有一个输入`in`，也只有一个输出`out`，其推导out tensor的形状的`ShapeInferFn`是一个lambda函数，参数为`InferContext`，返回值为`Maybe<void>`类型对象。

具体到`Relu` op对应的`ShapeInferFn`，其输出的形状与输入相同，即如上述代码所示。



## 实现Kernel
在写完Op的定义之后，我们需要为其提供一个或多个Kernel作为实现。此处我们实现一个支持在gpu上执行float数据类型的Kernel，存放在`relu_gpu.cu`文件中。

``` cpp
#include <cuda.h>
#include "oneflow/core/framework/framework.h"

template<typename T>
__global__ void ReluForwardGpu(const int n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = x[i] > 0 ? x[i] : 0; }
}

class ReluGpuFloatKernel final : public oneflow::user_op::OpKernel {
 public:
  ReluGpuFloatKernel(const oneflow::user_op::KernelInitContext& ctx) : oneflow::user_op::OpKernel(ctx) {}
  ~ReluGpuFloatKernel() = default;

 private:
  void Compute(oneflow::user_op::KernelContext* ctx) override {
    const oneflow::user_op::Tensor* in_blob = ctx->Tensor4ArgNameAndIndex("in", 0);
    oneflow::user_op::Tensor* out_blob = ctx->Tensor4ArgNameAndIndex("out", 0);

    int32_t n = in_blob->shape().elem_cnt();
    const float* in_ptr = in_blob->dptr<float>();
    float* out_ptr = out_blob->mut_dptr<float>();
    ReluForwardGpu<float><<<32, 1024, 0, ctx->device_ctx()->cuda_stream()>>>(n, in_ptr, out_ptr);
  }
};

REGISTER_USER_KERNEL("Relu")
    .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {
      return new ReluGpuFloatKernel(ctx);
    })
    .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {
      const user_op::TensorDesc* out_tensor = ctx.TensorDesc4ArgNameAndIndex("out", 0);
      return ctx.device() == oneflow::DeviceType::kGPU
             && out_tensor->data_type() == oneflow::DataType::kFloat;
    });
```

上述代码中，我们定义了一个类`ReluGpuFloatKernel`，继承自`oneflow::user_op::OpKernel`，构造函数参数为`oneflow::user_op::KernelInitContext`，且`override`了`Compute(KernelContext*)`函数。

在`Compute()`中，调用了`ReluForwardGpu()`这个CUDA Kernel来在GPU上执行Relu的逻辑。

除了书写`ReluGpuFloatKernel`类，我们还需要把`ReluGpuFloatKernel`注册到OneFlow中去。具体而言，就是书写两个lambda函数，分别是：

- `CreateFn`：输入`KernelInitContext`，返回要创建的Kernel即`ReluGpuFloatKernel`的指针。
- `IsMatchPred`：输入`KernelRegContext`，返回`bool`值，代表`Relu` op在什么情况下会选择`ReluGpuFloatKernel`。上述代码中，当设备类型为`GPU`且数据类型为`float`时，就选择`ReluGpuFloatKernel`。



对于其他设备类型（如`CPU`）或这其他数据类型（如`double`, `int`），其书写和注册Kernel的方式也是类似的。



## 编译Op的library (chengcheng)
  你可以通过`C++`的编译器(如 `g++`)编译你的自定义op代码（`relu.cpp`）。Oneflow的PIP包中包含了需要include的头文件目录以及库文件，这些文件的路径与你本地的操作系统和机器有关。你可以使用Oneflow的python库中的sysconfig模块得到它们（oneflow使用python3）。其中， `get_include()`可以得到本机的头文件目录路径，`get_lib()`可以得到本机的动态链接库路径，下面是在Linux机器上的输出结果：
```bash
$ python3
>>> import oneflow
>>> oneflow.sysconfig.get_include()
'/usr/local/lib/python3.6/site-packages/oneflow/include'
>>> oneflow.sysconfig.get_lib()
'/usr/local/lib/python3.6/site-packages/oneflow'
```
  oneflow.sysconfig库还包含编译选项和链接选项，可以帮助你编译生成你的自定义op动态库。假设你的Ubuntu系统上已经安装了`g++`，你可以使用下面的几行命令来生成你自己的动态库：
```bash
OF_CFLAGS=( $(python3 -c 'import oneflow; print(" ".join(oneflow.sysconfig.get_compile_flags()))') )
OF_LFLAGS=( $(python3 -c 'import oneflow; print(" ".join(oneflow.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared relu.cpp -o relu.so -fPIC ${OF_CFLAGS[@]} ${OF_LFLAGS[@]} -O2
```
### 编译支持GPU的Op library
  在上述的relu op的构建中，relu.cpp文件实现了`relu op`的注册，你也可以在该文件中添加relu kernel的CPU版本；`relu_gpu.cu`中实现了relu kernel的GPU-float版本。你可以使用以下命令编译你的gpu kernel的代码，并链接进你的自定义动态库中：
```bash
nvcc -std=c++11 -c -o relu_gpu.cu.o relu_gpu.cu ${OF_CFLAGS[@]}  -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o cuda_relu.so relu.cpp relu_gpu.cu.o  \
  ${OF_CFLAGS[@]} -fPIC -L/usr/local/cuda-10.0/lib64 -lcudart ${OF_LFLAGS[@]}
```
注意： 当你的环境中的CUDA**不是**安装在`/usr/local/lib64`目录下时，你需要在第二行g++的编译命令中指定CUDA的library目录。举个例子： 添加`-L /usr/local/cuda-10.0/lib64/`， 如果你的CUDA的安装目录是在`/usr/local/cuda-10.0/`。

## 在Python使用Op (chengcheng)

- python加载自定义op
Oneflow的python库提供`oneflow.config.load_library`函数来加载用户自定义的op动态库。该函数没有返回值。同时你需要编写一个函数表示该Op的python wrapper。Oneflow的python库提供了`user_op_builder`的类，用于生成一个op的wrapper。`user_op_builder`的使用方法见后续说明，对于简单的自定义my_op，可以参考下列python脚本使用和测试其正确性。
```python
import oneflow as flow
import numpy as np
# 默认配置
flow.config.gpu_device_num(1)
# 加载模块
flow.config.load_library("relu.so")
# python op wrapper function
def relu(input_blob, op_name):
  return flow.user_op_builder(op_name).Op("Relu").Input("in", [input_blob]).Build().RemoteBlobList()[0]

# 定义你的Job的配置
my_func_config = flow.FunctionConfig()                                                                 
my_func_config.default_distribute_strategy(flow.distribute.consistent_strategy())                      
my_func_config.default_data_type(flow.float) 
# 网络代码
@flow.function(my_func_config)
def MyJob(x = flow.FixedTensorDef((5,))):
  return relu(x, "my_relu_op_name")

# 执行
input_data = [-2,-1,0,1,2]
output_data = MyJob(np.array(input_data, dtype=np.float32)).get().ndarray()
print(output_data)

# 期望执行结果
[0. 0. 0. 1. 2.]
```
- 其他测试示例
对于复杂的自定义op单元测试或者网路测试，可以参考oneflow代码仓库中的`oneflow/python/test/ops`目录下的样例进行编写。
- python op wrapper
你可以使用`oneflow.user_op_builder`来生成你自定义op的python wrapper。`user_op_builder` 有一些特定规则来得到最终的输出blob：
  1. `user_op_builder("your_op_name")` 构造函数，参数为这个op的实际名字
  2. `.Op("op_type_name")` 指定这个op的type  必选项，只可调用一次
  3. `.Input("input_arg_name", input_blob_list)`  可选项，可以调用多次，每次指定一个`input_arg_name`，同时传入一个输入的blob列表，表示这个输入参数名字对应的多个输入blob
  4. `.Output("output_arg_name", num)` 可选项，可调用多次，每次指定一个`output_arg_name`实际对应的输出blob的数量，默认`num=1`。
  5. `.SetAttr("attr_name", attr_value, attr_type)`  可选项，可调用多次，每次指定一个attr属性的参数取值和参数类型，参数取值和类型的合法性由该op_def的attr属性判断。
  6. `.Build()` 只可调用一次，并且在上述属性指定结束后，该方法返回一个op的python wrapper
  7. `.RemoteBlobList()` 该方法是python op wrapper的接口，用于返回该op的输出blob的列表，列表中的每个blob中的`logical_blob_name`表示该blob是该op的哪个输出blob。列表的顺序是你在python wrapper中定义的Output()参数的顺序。当你有多个output_arg_name时，假设你定义的输出是`.Output("a", 2).Output("b",3)`，则RemoteBlobList()返回的列表长度为5，分别表示`[("a",0),("a",1),("b",0),("b",1),("b",2)]` 。

对于上述python示例中的relu函数的定义，可以有如下解释：
```python
def relu(input_blob, op_name):
  return flow.user_op_builder(op_name).Op("Relu") \
           .Input("in", [input_blob]) \
           .Output("out") \
           .Build().RemoteBlobList()[0]
# flow.user_op_builder(op_name)  -- 生成一个 op_name 作为其op的名称的op
#     .Op("Relu")             -- 这个op的类型是Relu
#     .Input("in", [input_blob]) -- 这个op有一个input_arg_name 为 "in"，并且对应的输入blob列表是[input_blob]，表示in只对应一个blob
#     .Output("out")             -- 这个op有一个output_arg_name 为 "out"，默认out对应的blob数量是1
#     .Build()                   -- 生成该op的python wrapper
#     .RemoteBlobList()[0]       -- 返回该op的唯一输出blob
```

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
| kAtListFloat   | std::vector< float >   |

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



#### Sbp Function、BatchAxis

Sbp、BatchAxis是OneFlow框架特有的概念，具体的请参见OneFlow文档。
`TODO()`

#### 性能优化： Inplace、KeepHeaderOnly
`TODO()`
#### is_mutable   mut消费input
`TODO()`
#### required_grad  定义input是否需要注册grad
`TODO()`
#### TODO...

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





### 注册Op的Grad (chengcheng)

- 说明及示例
Oneflow使用自动求导的方式进行后向计算图展开，为了对自定义的op进行后向求导，需要你注册一个后向生成函数来根据这个op的输出blob的导数计算输入blob的导数。你可以通过已有的其他op来构建这个后向展开的子图，当无法用已有op来描述后向时，你需要自己实现一个后向grad_op来表示。
后向生成函数在c++端注册。对于relu op，其后向生成函数的示例如下：
```cpp
#include "oneflow/core/framework/framework.h"

namespace oneflow {

REGISTER_USER_OP_GRAD("Relu").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                          user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    user_op::UserOpConfWrapper relu_grad_op =
        builder.Op("relu_grad")
            .Input("y", op.output("out", 0))
            .Input("dy", op.GetGradTensorWithOpOutput("out", 0))
            .Output("dx")
            .Build();
    op.BindGradTensorWithOpInput(relu_grad_op.output("dx", 0), "in", 0);
    AddOp(relu_grad_op);
  }
});

}  // namespace oneflow
```
- 后向生成函数的步骤：

宏`REGISTER_USER_OP_GRAD(op_type_name).SetGenBackwardOpConfFn(fn);`用来注册你的自定义op的后向生成函数，其中fn函数具有两个参数，`UserOpWrapper op`和`AddOpFn AddOp`，其中op表示你的自定义op，AddOp表示向整个计算图中添加一个新的op（用于后向图展开）。
在生成后向图的函数中，图里的blob是由一个叫logical blob name的字符串表示的，其中包含了产生这个blob的op的name，以及这个blob在这个op里的name。后向生成函数的任务是对前向op的输入blob，生成一个op的子图，这个子图接收前向op的输入输出blob以及输出blob的导数（梯度/grad）blob，子图的最终输出是前向op输入blob对应的导数（梯度）blob，因此针对每个（可能）需要生成梯度的blob，都需要构建一个由其他op组成的子图，并将子图的输出blob与这个需要生成梯度的blob绑定。编写这个生成子图的过程通常包含下面几步：
  1. 判断前向op的某一个input blob是否需要生成后向的梯度blob。（我们强烈建议进行这个判断，即使你认为这个op的input一定会有梯度。因为oneflow在系统的构图优化后，可能会分析得到这个input的梯度计算是没有意义的，则可以知道不需要构建这个额外的后向子图。）
  2. 使用UserOpConfWrapperBuilder来构建这个子图中的new_op，通常这些new_op的输入是前向op的in/out或者out对应的out_grad。
  3. 将构建好的子图的输出blob的logical blob name与前向op的输入blob绑定
  4. 使用AddOp函数将第2步中构建的new_op添加到计算图中

针对上图中的relu_op的例子，我们对应一下每个步骤：
```cpp
REGISTER_USER_OP_GRAD("Relu").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                          user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {   /* step 1. 判断relu_op.in(0) 是否需要构建后向子图*/
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    /* step 2. 使用UserOpConfWrapperBuilder来构建子图，该子图中包含一个 relu_grad_op */
    user_op::UserOpConfWrapper relu_grad_op =
        builder.Op("relu_grad")
            .Input("y", op.output("out", 0)) /* relu grad op 的一个输入是y，对应的blob为前向op的out(0) */
            .Input("dy", op.GetGradTensorWithOpOutput("out", 0))  /* 另一个输入dy对应的blob为out_grad */
            .Output("dx")
            .Build();
    /* step 3. 绑定子图的输出blob（relu_grad.out("dx")）与 前向op的input grad blob */
    op.BindGradTensorWithOpInput(relu_grad_op.output("dx", 0), "in", 0);
    AddOp(relu_grad_op);  /* step 4. 将子图中新创建的op添加到计算图中 */
  }
});
```
- 可能用到的接口介绍：
  1. `UserOpWrapper`:
    `.NeedGenGradTensor4OpInput(input_arg_name, index)` 返回一个bool值，判断前向op的输入是否需要生成后向的梯度
    `.input(arg_name,index)` 得到输入的logical blob name
    `.output(arg_name,index)` 得到输出的logical blob name
    `.attr(attr_name)` 得到op的属性值
    `.TensorDesc4ArgNameAndIndex(arg_name, index)` 返回前向op的输入/输出对应的TensorDesc，包含shape、dtype信息
    `.GetGradTensorWithOpOutput(output_arg_name, index)` 返回前向op的输出对应的后向梯度blob的logical blob name
    `.BindGradTensorWithOpInput(logical_blob_name, input_arg_name, index)` 将一个特定的logical blob name与该前向op的输入梯度blob绑定
  2. `UserOpConfWrapperBuilder`:  （与python端的user_op_builder功能一致，接口类似）
    `UserOpConfWrapperBuilder(your_op_name)`  构造函数需要输入新构建的op name
    `.Op(op_type_name)`  指定这个op的type
    `.Input(arg_name, logical_blob_name)`  可选项，可以调用多次，每次指定一个input_arg_name，同时传入一个logical_blob_name，表明这个input arg name对应的blob。如果该input_arg_name对应多个输入blob，则调用`.Input()`的顺序就是其对应的index顺序
    `.Output(arg_name, num)`  可选项，可调用多次，每次指定一个`output_arg_name`实际对应的输出blob的数量，也可以调用 `.Output(arg_name)`，表示`num = 1`
    `.Attr(attr_name, val)` 可选项，可调用多次，每次指定一个attr属性的属性名称和参数值，表示对这个attr赋值为val
    `.Build()`  返回一个UserOpConfWrapper，表示你构建完毕的新op
  3. `UserOpConfWrapper`:
    `.input(arg_name,index)` 得到输入的logical blob name
    `.output(arg_name,index)` 得到输出的logical blob name
    `.attr(attr_name)` 得到op的属性值
  4. `AddOp`: 
    输入参数是一个UserOpConfWrapper
