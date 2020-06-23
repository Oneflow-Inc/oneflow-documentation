# �����µ�Op

## ����
���ĵ���������Щ��Ҫ�Զ���ʵ��C++ Op���û���ͨ����˵������Դ����������

- ����ͨ��OneFlow���е�Op��Pythonǰ����ϴ������ɡ�
- ����ͨ��OneFlow����Op��Pythonǰ����ϴ������ɣ���ȴ���㲻����������
- �û���Ҫ�ֶ���kernel�����ں�fuse��������ĳЩ�ض�������

����������֮������������Ƽ�����Pythonǰ��ʹ������Op��ϵõ���Ҫ��Op��



�ڽ����Զ���Op֮ǰ�����Ǽ�Ҫ���������ѧϰ����еļ������̺Ͷ�Ӧ��Op�Լ�Kernel�Ĺ�ϵ��

�Լ����ReluΪ����������������Relu����󣬵õ�����ֵ���������һ���������̾ͳ�Ϊһ��Op�����Op�Ƚϼ򵥣�ֻ�����һ��relu�����ļ��㼴�ɣ����ǿ��ԶԴ�Op��������4�����͵�Kernelʵ��:

- gpu�µ�float���͵�kernel

- gpu�µ�int���͵�kernel

- cpu�µ�float���͵�kernel

- cpu�µ�int���͵�kernel

ͬ���ĵ�������Batch Normal��BN)��ʵ�֣����ܾ͸��ӵ㣬BN���̶���ÿ�����룬�����¼�����̣�

![[��ʽ]](https://www.zhihu.com/equation?tex=%5Cmu_j%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em+Z_j%5E%7B%28i%29%7D)

![[��ʽ]](https://www.zhihu.com/equation?tex=%5Csigma%5E2_j%3D%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5Em%28Z_j%5E%7B%28i%29%7D-%5Cmu_j%29%5E2)

![[��ʽ]](https://www.zhihu.com/equation?tex=%5Chat%7BZ%7D_j%3D%5Cfrac%7BZ_j-%5Cmu_j%7D%7B%5Csqrt%7B%5Csigma_j%5E2%2B%5Cepsilon%7D%7D)

���Կ���������ɷ�Ϊ3�����裬���ÿ��������Ҫ֮ͬǰRelu���Ƶ�kernel��֧��cpu��gpu�µ�int��float���ͣ������ܹ���Ҫ3��4 = 12��kernel�����⻹û���ǵ�ǰ��ͷ��򴫲�ʱBN���㷽ʽ�Ĳ�ͬ��



�ɴ˿ɼ����Զ���Op�Ŀ����ǱȽϸ��ӵ�һ���£���Ϊ�䲻����c++��python����Ŀ��������漰�����ѧϰ�ļ��������Լ�ҵ��������Ǳ�Ҫ������ʹ��OneFlow���е�Op��

���ڵ�ȷ��Ҫ�ֶ�ʵ��C++ Op���û�������Ҫ���²��������Զ���Op����������

1. ����ע��Op�� Op�Ķ��������Op��ʵ�֣���Kernel������������Op�Ĺ����ԡ�ͨ������Op�����ƣ�Op������������Op���������Ժ�һЩ��Ҫ�������Ƶ�Tensor��shape��data type�ĺ�����

2. ʵ��Op��Ӧ��Kernel�� ��C++����ʵ��Op��Ӧ��Kernel��Kernelͨ����������Op����ϸ������̣�����һ��Op��˵�����ܻ��Ӧ���Kernel��Kernel��д��ɺ���Ҫע��������󶨵�Op��

3. ʵ��Op��Python���ô��룺��ΪOneFlow�����繹���ǻ���Pythonȥ��д�ģ�����������Ҫ��Pythonǰ��ȥ��д������������װǰ������д��C++�Ĵ��롣

4. ����ѡ��ע��Op��Ӧ�ĺ��򴫲����������ѵ����������ҪOp�ĺ��򴫲�����ô���ǻ���Ҫȥдһ������������OneFlow�����ȥ������Op��Ӧ�ĺ��������̡�

5. ����Op����������ȫ��ִ����Ϻ����ǻ���Ҫ��Op���в����Ա�֤Op����ȷ�ԣ����岽���������

   


## �����µ�Op

������`Relu`Ϊ��������һ��Opע������̣�����һ��cpp�ļ�`relu.cpp`����ע��Op��

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

�������������룺

������include��`oneflow/core/framework/framework.h`��`framework.h`�а��������Ǵ���һ��Op��Ҫ������ͷ�ļ�����������ֻ��Ҫinclude��һ��ͷ�ļ����С�

��Σ�Ϊ�˼���������������`namespace oneflow`����дע��Op�������ش��롣

�������������ע��Ĵ��롣`Relu`  opֻ��һ������`in`��Ҳֻ��һ�����`out`�����Ƶ�out tensor����״��`ShapeInferFn`��һ��lambda����������Ϊ`InferContext`������ֵΪ`Maybe<void>`���Ͷ���

���嵽`Relu` op��Ӧ��`ShapeInferFn`�����������״��������ͬ����������������ʾ��



## ʵ��Kernel
��д��Op�Ķ���֮��������ҪΪ���ṩһ������Kernel��Ϊʵ�֡��˴�����ʵ��һ��֧����gpu��ִ��float�������͵�Kernel�������`relu_gpu.cu`�ļ��С�

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

���������У����Ƕ�����һ����`ReluGpuFloatKernel`���̳���`oneflow::user_op::OpKernel`�����캯������Ϊ`oneflow::user_op::KernelInitContext`����`override`��`Compute(KernelContext*)`������

��`Compute()`�У�������`ReluForwardGpu()`���CUDA Kernel����GPU��ִ��Relu���߼���

������д`ReluGpuFloatKernel`�࣬���ǻ���Ҫ��`ReluGpuFloatKernel`ע�ᵽOneFlow��ȥ��������ԣ�������д����lambda�������ֱ��ǣ�

- `CreateFn`������`KernelInitContext`������Ҫ������Kernel��`ReluGpuFloatKernel`��ָ�롣
- `IsMatchPred`������`KernelRegContext`������`bool`ֵ������`Relu` op��ʲô����»�ѡ��`ReluGpuFloatKernel`�����������У����豸����Ϊ`GPU`����������Ϊ`float`ʱ����ѡ��`ReluGpuFloatKernel`��



���������豸���ͣ���`CPU`�����������������ͣ���`double`, `int`��������д��ע��Kernel�ķ�ʽҲ�����Ƶġ�



## ����Op��library (chengcheng)
  �����ͨ��`C++`�ı�����(�� `g++`)��������Զ���op���루`relu.cpp`����Oneflow��PIP���а�������Ҫinclude��ͷ�ļ�Ŀ¼�Լ����ļ�����Щ�ļ���·�����㱾�صĲ���ϵͳ�ͻ����йء������ʹ��Oneflow��python���е�sysconfigģ��õ����ǣ�oneflowʹ��python3�������У� `get_include()`���Եõ�������ͷ�ļ�Ŀ¼·����`get_lib()`���Եõ������Ķ�̬���ӿ�·������������Linux�����ϵ���������
```bash
$ python3
>>> import oneflow
>>> oneflow.sysconfig.get_include()
'/usr/local/lib/python3.6/site-packages/oneflow/include'
>>> oneflow.sysconfig.get_lib()
'/usr/local/lib/python3.6/site-packages/oneflow'
```
  oneflow.sysconfig�⻹��������ѡ�������ѡ����԰����������������Զ���op��̬�⡣�������Ubuntuϵͳ���Ѿ���װ��`g++`�������ʹ������ļ����������������Լ��Ķ�̬�⣺
```bash
OF_CFLAGS=( $(python3 -c 'import oneflow; print(" ".join(oneflow.sysconfig.get_compile_flags()))') )
OF_LFLAGS=( $(python3 -c 'import oneflow; print(" ".join(oneflow.sysconfig.get_link_flags()))') )
g++ -std=c++11 -shared relu.cpp -o relu.so -fPIC ${OF_CFLAGS[@]} ${OF_LFLAGS[@]} -O2
```
### ����֧��GPU��Op library
  ��������relu op�Ĺ����У�relu.cpp�ļ�ʵ����`relu op`��ע�ᣬ��Ҳ�����ڸ��ļ������relu kernel��CPU�汾��`relu_gpu.cu`��ʵ����relu kernel��GPU-float�汾�������ʹ����������������gpu kernel�Ĵ��룬�����ӽ�����Զ��嶯̬���У�
```bash
nvcc -std=c++11 -c -o relu_gpu.cu.o relu_gpu.cu ${OF_CFLAGS[@]}  -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o cuda_relu.so relu.cpp relu_gpu.cu.o  \
  ${OF_CFLAGS[@]} -fPIC -L/usr/local/cuda-10.0/lib64 -lcudart ${OF_LFLAGS[@]}
```
ע�⣺ ����Ļ����е�CUDA**����**��װ��`/usr/local/lib64`Ŀ¼��ʱ������Ҫ�ڵڶ���g++�ı���������ָ��CUDA��libraryĿ¼���ٸ����ӣ� ���`-L /usr/local/cuda-10.0/lib64/`�� ������CUDA�İ�װĿ¼����`/usr/local/cuda-10.0/`��

## ��Pythonʹ��Op (chengcheng)

- python�����Զ���op
Oneflow��python���ṩ`oneflow.config.load_library`�����������û��Զ����op��̬�⡣�ú���û�з���ֵ��ͬʱ����Ҫ��дһ��������ʾ��Op��python wrapper��Oneflow��python���ṩ��`user_op_builder`���࣬��������һ��op��wrapper��`user_op_builder`��ʹ�÷���������˵�������ڼ򵥵��Զ���my_op�����Բο�����python�ű�ʹ�úͲ�������ȷ�ԡ�
```python
import oneflow as flow
import numpy as np
# Ĭ������
flow.config.gpu_device_num(1)
# ����ģ��
flow.config.load_library("relu.so")
# python op wrapper function
def relu(input_blob, op_name):
  return flow.user_op_builder(op_name).Op("Relu").Input("in", [input_blob]).Build().RemoteBlobList()[0]

# �������Job������
my_func_config = flow.FunctionConfig()                                                                 
my_func_config.default_distribute_strategy(flow.distribute.consistent_strategy())                      
my_func_config.default_data_type(flow.float) 
# �������
@flow.function(my_func_config)
def MyJob(x = flow.FixedTensorDef((5,))):
  return relu(x, "my_relu_op_name")

# ִ��
input_data = [-2,-1,0,1,2]
output_data = MyJob(np.array(input_data, dtype=np.float32)).get().ndarray()
print(output_data)

# ����ִ�н��
[0. 0. 0. 1. 2.]
```
- ��������ʾ��
���ڸ��ӵ��Զ���op��Ԫ���Ի�����·���ԣ����Բο�oneflow����ֿ��е�`oneflow/python/test/ops`Ŀ¼�µ��������б�д��
- python op wrapper
�����ʹ��`oneflow.user_op_builder`���������Զ���op��python wrapper��`user_op_builder` ��һЩ�ض��������õ����յ����blob��
  1. `user_op_builder("your_op_name")` ���캯��������Ϊ���op��ʵ������
  2. `.Op("op_type_name")` ָ�����op��type  ��ѡ�ֻ�ɵ���һ��
  3. `.Input("input_arg_name", input_blob_list)`  ��ѡ����Ե��ö�Σ�ÿ��ָ��һ��`input_arg_name`��ͬʱ����һ�������blob�б���ʾ�������������ֶ�Ӧ�Ķ������blob
  4. `.Output("output_arg_name", num)` ��ѡ��ɵ��ö�Σ�ÿ��ָ��һ��`output_arg_name`ʵ�ʶ�Ӧ�����blob��������Ĭ��`num=1`��
  5. `.SetAttr("attr_name", attr_value, attr_type)`  ��ѡ��ɵ��ö�Σ�ÿ��ָ��һ��attr���ԵĲ���ȡֵ�Ͳ������ͣ�����ȡֵ�����͵ĺϷ����ɸ�op_def��attr�����жϡ�
  6. `.Build()` ֻ�ɵ���һ�Σ���������������ָ�������󣬸÷�������һ��op��python wrapper
  7. `.RemoteBlobList()` �÷�����python op wrapper�Ľӿڣ����ڷ��ظ�op�����blob���б��б��е�ÿ��blob�е�`logical_blob_name`��ʾ��blob�Ǹ�op���ĸ����blob���б��˳��������python wrapper�ж����Output()������˳�򡣵����ж��output_arg_nameʱ�������㶨��������`.Output("a", 2).Output("b",3)`����RemoteBlobList()���ص��б���Ϊ5���ֱ��ʾ`[("a",0),("a",1),("b",0),("b",1),("b",2)]` ��

��������pythonʾ���е�relu�����Ķ��壬���������½��ͣ�
```python
def relu(input_blob, op_name):
  return flow.user_op_builder(op_name).Op("Relu") \
           .Input("in", [input_blob]) \
           .Output("out") \
           .Build().RemoteBlobList()[0]
# flow.user_op_builder(op_name)  -- ����һ�� op_name ��Ϊ��op�����Ƶ�op
#     .Op("Relu")             -- ���op��������Relu
#     .Input("in", [input_blob]) -- ���op��һ��input_arg_name Ϊ "in"�����Ҷ�Ӧ������blob�б���[input_blob]����ʾinֻ��Ӧһ��blob
#     .Output("out")             -- ���op��һ��output_arg_name Ϊ "out"��Ĭ��out��Ӧ��blob������1
#     .Build()                   -- ���ɸ�op��python wrapper
#     .RemoteBlobList()[0]       -- ���ظ�op��Ψһ���blob
```

## �߼�����
������Ϊֹ�������Ѿ����`Relu` op�Ĺ�������Ȼ��Relu Op��һ���Ƚϼ򵥵�Op�����������Ҫ����һ���Ƚϸ��ӵ�Op������Ҫʹ��һЩ����ĸ߼�������Э�����ǡ�

### Op Registration

#### Attribute

�е�Op��Ҫ���������ԣ�����`Conv` op��Ҫ������`padding`�ķ�ʽ��`Reshape` op��Ҫ����һ��`tensor shape`����Op����ӵ�Graph��ʱ�����Ǿ���Ҫ����Щ�������ú����ֵ�ˡ�

��OneFlow�У��������ע��Opʱָ������Ҫ������Attr���˴�������`Reshape`  opΪ����

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

���������У�����Ϊ`Reshape` op������һ��attribute������Ϊ`shape`����������`UserOpAttrType::kAtShape`��

��OneFlow�У�����Ŀǰ֧�������¼���AttrType��

| UserOpAttrType | ��Ӧ��C++��������    |
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

��`Reshape` opע���`ShapeInferFn`�У���Ͱ�`shape`��Ӧ��Attr��ֵ��`oneflow::Shape`���ͣ�������`out` tensor��Ӧ��`out_shape`��



����ָ��Attr�����ͣ����ǻ�����Ϊ������һ��Ĭ��ֵ��Ĭ��ֵ�����ͼ�����ж�Ӧ��C++�������ͣ��磺

``` cpp
.Attr("is_transpose", UserOpAttrType::kAtBool, false)
    
.Attr("size", UserOpAttrType::kAtInt32, 10)
    
.Attr("vector_of_size", UserOpAttrType::kAtListInt32, std::vector<int32_t>{10, 11, 12})
```



#### Check Attribute Function

����ĳЩAttribute��˵������Ҫ����ϸ�Ļ���ȡֵ��Χ����ʱ����Ҫ��ע��Opʱͨ��`CheckAttrFn`��ָ����ȡֵ��Χ��

���磬����`Conv` op��˵������һ������ѡ��`data_format`����������string�ַ�������ȡֵֻ����`channels_first`��`channels_last`������������֮�ⶼ���Ϸ�������������Ҫָ���䷶Χ��ע��Opʱ����Ҫ����ָ����

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

������ЩOp��˵���乲��ͬһ��name�� in/out tensor�����ж�����������`Add` op��˵����`input`��� name�¿��ܶ�Ӧ�ж��tensor����ʱ���Ǿ���Ҫ��ע��Opʱָ�����Ӧ����������ĸ�����

OneFlow���֧�ֶ�Op��Input/Output���������ã�

```cpp
.Input("input") // input �����Ӧ��1��tensor

.Input("input", 5) // input �����Ӧ��5��tensor
    
.InputWithMinimum("input", 5) // input �����Ӧ����5��tensor
    
.OptionalInput("input") // input ����û�ж�Ӧ��tensor�����������Ӧ1��tensor
    
.OptionalInput("input", 5) // input ����û�ж�Ӧ��tensor�����������Ӧ5��tensor
    
.OptionalInputWithMininum("input", 5) // input ����û�ж�Ӧ��tensor�����������Ӧ����5��tensor
    
    
// Output��Input�÷���ͬ
```



#### DataType Infer Function

����Op��input tensor��output tensor��������ͬ��������һЩ����Op����`Cast` op����˵������Ҫ����һ��data_type infer function���Ƶ�output tensor�����͡�

``` cpp
.SetDataTypeInferFn([](user_op::InferContext* ctx) {
      DataType* out_tensor_type = ctx->Dtype4ArgNameAndIndex("out", 0);
      *out_tensor_type = DataType::kDouble;
      return Maybe<void>::Ok();
    })
```

����������ǰ�out tensor��������������Ϊ`double`��

������������������͵�Op��Ҳ������ע��Opʱָ��`SetDataTypeInferFn()`����ΪOneFlow����ṩ��Ĭ��ʵ�־�����output tensors �� input tensors ����������һ�¡�



#### Sbp Function��BatchAxis

Sbp��BatchAxis��OneFlow������еĸ���������μ�OneFlow�ĵ���
`TODO()`

#### �����Ż��� Inplace��KeepHeaderOnly
`TODO()`
#### is_mutable   mut����input
`TODO()`
#### required_grad  ����input�Ƿ���Ҫע��grad
`TODO()`
#### TODO...

### Kernel Registration

#### Temporary Buffer Size Infer Function

����һЩOp��˵����ĳ��ʵ�֣���Kernel�����ܻ���ҪһЩ�����buffer���洢һЩ��ʱ���ݡ�

��OneFlow����У���Щ��ʱ��bufferҲ����Ϊtensor����`Compute`������ʹ�õġ�

����Ҫ������ʱbuffer���͵�������ע��Kernelʱָ���ˡ�

``` cpp
REGISTER_USER_KERNEL("XOp")
    .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) { return new XKernel(ctx); })
    .SetIsMatchedPred(...)
    .SetInferTmpSizeFn([](const oneflow::user_op::InferContext*) { return 1024; });
// XOp ��Ӧ�� XKernel ��Ҫ1024Byte��С��buffer

class XKernel final : public oneflow::user_op::OpKernel {
...
  void Compute(oneflow::user_op::KernelContext* ctx) override {
    ...
    oneflow::user_op::Tensor* tmp = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    // ����1024Byte��tensor
    ...
  }
};
```





### ע��Op��Grad (chengcheng)

- ˵����ʾ��
Oneflowʹ���Զ��󵼵ķ�ʽ���к������ͼչ����Ϊ�˶��Զ����op���к����󵼣���Ҫ��ע��һ���������ɺ������������op�����blob�ĵ�����������blob�ĵ����������ͨ�����е�����op�������������չ������ͼ�����޷�������op����������ʱ������Ҫ�Լ�ʵ��һ������grad_op����ʾ��
�������ɺ�����c++��ע�ᡣ����relu op����������ɺ�����ʾ�����£�
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
- �������ɺ����Ĳ��裺

��`REGISTER_USER_OP_GRAD(op_type_name).SetGenBackwardOpConfFn(fn);`����ע������Զ���op�ĺ������ɺ���������fn������������������`UserOpWrapper op`��`AddOpFn AddOp`������op��ʾ����Զ���op��AddOp��ʾ����������ͼ�����һ���µ�op�����ں���ͼչ������
�����ɺ���ͼ�ĺ����У�ͼ���blob����һ����logical blob name���ַ�����ʾ�ģ����а����˲������blob��op��name���Լ����blob�����op���name���������ɺ����������Ƕ�ǰ��op������blob������һ��op����ͼ�������ͼ����ǰ��op���������blob�Լ����blob�ĵ������ݶ�/grad��blob����ͼ�����������ǰ��op����blob��Ӧ�ĵ������ݶȣ�blob��������ÿ�������ܣ���Ҫ�����ݶȵ�blob������Ҫ����һ��������op��ɵ���ͼ��������ͼ�����blob�������Ҫ�����ݶȵ�blob�󶨡���д���������ͼ�Ĺ���ͨ���������漸����
  1. �ж�ǰ��op��ĳһ��input blob�Ƿ���Ҫ���ɺ�����ݶ�blob��������ǿ�ҽ����������жϣ���ʹ����Ϊ���op��inputһ�������ݶȡ���Ϊoneflow��ϵͳ�Ĺ�ͼ�Ż��󣬿��ܻ�����õ����input���ݶȼ�����û������ģ������֪������Ҫ�����������ĺ�����ͼ����
  2. ʹ��UserOpConfWrapperBuilder�����������ͼ�е�new_op��ͨ����Щnew_op��������ǰ��op��in/out����out��Ӧ��out_grad��
  3. �������õ���ͼ�����blob��logical blob name��ǰ��op������blob��
  4. ʹ��AddOp��������2���й�����new_op��ӵ�����ͼ��

�����ͼ�е�relu_op�����ӣ����Ƕ�Ӧһ��ÿ�����裺
```cpp
REGISTER_USER_OP_GRAD("Relu").SetGenBackwardOpConfFn([](const user_op::UserOpWrapper& op,
                                                          user_op::AddOpFn AddOp) {
  if (op.NeedGenGradTensor4OpInput("in", 0)) {   /* step 1. �ж�relu_op.in(0) �Ƿ���Ҫ����������ͼ*/
    user_op::UserOpConfWrapperBuilder builder(op.op_name() + "_grad");
    /* step 2. ʹ��UserOpConfWrapperBuilder��������ͼ������ͼ�а���һ�� relu_grad_op */
    user_op::UserOpConfWrapper relu_grad_op =
        builder.Op("relu_grad")
            .Input("y", op.output("out", 0)) /* relu grad op ��һ��������y����Ӧ��blobΪǰ��op��out(0) */
            .Input("dy", op.GetGradTensorWithOpOutput("out", 0))  /* ��һ������dy��Ӧ��blobΪout_grad */
            .Output("dx")
            .Build();
    /* step 3. ����ͼ�����blob��relu_grad.out("dx")���� ǰ��op��input grad blob */
    op.BindGradTensorWithOpInput(relu_grad_op.output("dx", 0), "in", 0);
    AddOp(relu_grad_op);  /* step 4. ����ͼ���´�����op��ӵ�����ͼ�� */
  }
});
```
- �����õ��Ľӿڽ��ܣ�
  1. `UserOpWrapper`:
      `.NeedGenGradTensor4OpInput(input_arg_name, index)` ����һ��boolֵ���ж�ǰ��op�������Ƿ���Ҫ���ɺ�����ݶ�
      `.input(arg_name,index)` �õ������logical blob name
      `.output(arg_name,index)` �õ������logical blob name
      `.attr(attr_name)` �õ�op������ֵ
      `.TensorDesc4ArgNameAndIndex(arg_name, index)` ����ǰ��op������/�����Ӧ��TensorDesc������shape��dtype��Ϣ
      `.GetGradTensorWithOpOutput(output_arg_name, index)` ����ǰ��op�������Ӧ�ĺ����ݶ�blob��logical blob name
      `.BindGradTensorWithOpInput(logical_blob_name, input_arg_name, index)` ��һ���ض���logical blob name���ǰ��op�������ݶ�blob��
  2. `UserOpConfWrapperBuilder`:  ����python�˵�user_op_builder����һ�£��ӿ����ƣ�
      `UserOpConfWrapperBuilder(your_op_name)`  ���캯����Ҫ�����¹�����op name
      `.Op(op_type_name)`  ָ�����op��type
      `.Input(arg_name, logical_blob_name)`  ��ѡ����Ե��ö�Σ�ÿ��ָ��һ��input_arg_name��ͬʱ����һ��logical_blob_name���������input arg name��Ӧ��blob�������input_arg_name��Ӧ�������blob�������`.Input()`��˳��������Ӧ��index˳��
      `.Output(arg_name, num)`  ��ѡ��ɵ��ö�Σ�ÿ��ָ��һ��`output_arg_name`ʵ�ʶ�Ӧ�����blob��������Ҳ���Ե��� `.Output(arg_name)`����ʾ`num = 1`
      `.Attr(attr_name, val)` ��ѡ��ɵ��ö�Σ�ÿ��ָ��һ��attr���Ե��������ƺͲ���ֵ����ʾ�����attr��ֵΪval
      `.Build()`  ����һ��UserOpConfWrapper����ʾ�㹹����ϵ���op
  3. `UserOpConfWrapper`:
      `.input(arg_name,index)` �õ������logical blob name
      `.output(arg_name,index)` �õ������logical blob name
      `.attr(attr_name)` �õ�op������ֵ
  4. `AddOp`: 
      ���������һ��UserOpConfWrapper
