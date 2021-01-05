# Extending Op with Python
**Attention** : The Python Kernel covered in this article was only fully tested under the `gcc 4.8.5` build environment. Further refinements are planned in [Issue 3951](https://github.com/Oneflow-Inc/oneflow/issues/3951).

## Background Information
OneFlow abstracts all kinds of data processing into operators. Op is act on the input tensor and writes the result of the calculation to the output tensor. All operators available of OneFlow  can be found in the [ops directory](https://github.com/Oneflow-Inc/oneflow/tree/master/oneflow/python/ops).

When OneFlow's existing Python operators and their combinations are not sufficient for building neural networks or when the Python level operators do not meet performance requirements. Then we can develop custom ops. OneFlow provides two ways to develop custom Ops. One is the Python based `Python Kernel` and the other is the `C++ Kernel` which is introduced in the article [Extending Op with C++](./user_op.md)

`Python Kernel` has a simple development process and is suitable for pre-research, algorithm verification and other scenarios because it is mainly extended by Python. The `C++ Kernel` is efficient and suitable for developing operators that have been proven reliability and with performance requirement.

This article introduces the background knowledge and basic concepts of operator development then demonstrates how to develop the `Python Kernel`.

### Basic Concepts
The concepts of `op_type_name`, `Op` and `Kernel` need to be understood before OneFlow operator development.

- op_type_name：op_type_name is the globally unique ID of op. op_type_name is used by OneFlow to confirm op type and then to instantiate op which is used to build the calculation map. The relationship between op type and op is similar to the relationship between class and object.
- op：Logical operators that contain information such as input and output shapes for reasoning. But do not contain specific computing logic for processing data.
- kernel：The computing logic of op could be different depending on the physical device and data type. The specific processing logic at runtime is handled by kernel. In brief, op has a one-to-many relationship with the kernels and we can use Python to do the specific operations which are called the `Python Kernel` or [Extending Op with C++](./user_op.md).
- The OneFlow kernel is implemented by C++ but the user interface uses Python. So you need to write the `Python Wrapper` conventionally to enable the Python Op interface to interact with the C++ kernel.

### Development Steps
To develop custom Op with Python, we should prepare a directory named `op_type_name` where you can place the necessary files in it.  Use [oneflow/python/test/custom_ops/user_sigmoid](https://github.com/Oneflow-Inc/oneflow/tree/master/oneflow/python/test/custom_ops/user_sigmoid) as example:

```text
user_sigmoid
├── user_sigmoid_cpp_def.cpp
├── user_sigmoid_py_api.py
└── user_sigmoid_py_kernel.py
```

In details:

- `op_type_name_cpp_def.cpp`(the  `user_sigmoid_cpp_def.cpp` above) store the definition of Op.
- `op_type_name_py_api.py`(the `user_sigmoid_py_api.py` above) store `Python Wrapper` which export the implemented `Python Kernel` to the user by using `oneflow.user_op_builder`.
- `op_type_name_py_kernel.py`(the `user_sigmoid_py_kernel.py` above) stores forward and backward computing logic for custom operators implemented by Python.


In the following section, we'll show how to implement a custom `user_relu` Op in Python which include:

- How to write `op_type_name_cpp_def.cpp` to define Op.
- How to write `op_type_name_py_api.py` to encapsulate the Python interface of Op.
- How to write the `op_type_name_py_kernel.py` and uses Python to implement Op's calculation Kernel.
- How to use `Python Kernel` custom Op in OneFlow.


## Implementation and Registration of Op
First, we define op in `user_relu_cpp_def.cpp` and register it.
```cpp
#include "oneflow/core/framework/framework.h"

namespace oneflow {
namespace {

REGISTER_USER_OP("user_relu_forward")
  .Attr<std::string>("device_sub_tag", "py")
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

Analyze the above code: 

- The `oneflow/core/framework/framework.h` contains all the interfaces we need to create an Op
- `.Attr<std::string>("device_sub_tag", "py")` is necessary which tells OneFlow to call the Python Kernel when using this Op.
- The interface related to the custom op is organized in `oneflow::user_op`. Using the namespace `oneflow` to simplify the type name.
- The macro `REGISTER_USER_OP` is used to register Op which accepts an parameter `user_relu_forward` as `op_type_name`.
- After registering with `REGISTER_USER_OP`, it actually returns an `OpRegistry` class (located in [user_op_registry.h](https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/core/framework/user_op_registry.h)). By calling this class, the setting of a custom op is accomplished:
    1. `Input("in") ` indicates it has an input named "in".
    2. `Output("out") ` indicates it has an output named "out".
    3. `SetTensorDescInferFn` is used to set the shape and data type inference function. Also to describe the  the shape and data type's relationship between the output and input. In the above code, the output shape and data type are as same as the input.

The `op_type_name_cpp_def.cpp` is the only C++ file that will be used in the implementation of the `Python Kernel`. It is used to configure information of Op. By far, we cannot remove the C++ configuration of Op since it is necessary for setting advanced information such as distributions. But as we can see, that file does not involve specific operations, so, even you are not familiar with C++, you can easily master it according to our examples.

## Wrapping Python Interface of Op
In order to make it easier for user to use the `user_relu` we just created at the Python level. We need to create a `user_relu_py_api.py` file with the following contents:

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

`flow.user_op_builder("op_myrelu")` actually returns a `UserOpConfBuilder` object named `op_myrelu`.

This object contains methods such as `Op`, `Input` and so on which are used to encapsulate a custom op e explained below:

- `Op("user_relu_forward")`：The parameter must be `op_type_name` which previously registered in C++. OneFlow finds the Op type already registered and instantiates the op object.
- `Input("in", [input_blob])`：Corresponds to `Input` for op registration in C++. The first parameter string must match the string set in `Input` when the op is registered in C++. The second parameter is the tensor of the input which is a `list` because an Op allows multiple inputs.
- `Output("out")`：This corresponds to `Output` for op registration in C++.
- `Build`：After the above settings, call `Build` to get the Python wrapper for the custom op

The following code will get the output of a custom Op:
```python
return op.InferAndTryRun().SoleOutputBlob()
```

The `InferAndTryRun` completes the inference and returns `UserOp`. If the result is the sole output, use `SoleOutputBlob` to get the unique output. Otherwise use `RemoteBlobList` to get a list with multiple outputs.

## Implementing the Kernel with Python
As described at the beginning of this article, Op is only a logical concept and the real computation need to be done in the Kernel. In OneFlow, it can be implemented in both C++ and Python.This article only describes the accessible implementations of the Python Kernel.  To implement the Kernel in C++, please refer to [Extending Op with C++](./user_op.md).

In order to provide the Python Kernel for the `user_relu` we set up previously, we need to create a `user_relu_py_kernel.py` file with the following contents:

```python
import numpy as np

def forward(args):
    (x,) = args
    y = (x>0)*x
    return y
```

The above `forward` method is necessary and its implementation corresponds to our Python Kernel of Op:

- The method name must be `forward`.
- There is only one parameter of type `tuple`. The number and order of elements in `tuple` corresponds to the `Input` of the Op registration. For example, we previously registered `Input("in")` for `user_relu`. Then `x` in `(x, ) = args` in the above code will take the value of `in`.
- Output corresponds to `Output` when Op is registered.
- Both parameters and return values are `numpy` objects which also means they cannot be strings, integers and etc.

## Using Custom Op
After finish above, we have a directory named `user_relu` which containing three files with the following structure:

```text
user_relu/
├── user_relu_cpp_def.cpp
├── user_relu_py_api.py
└── user_relu_py_kernel.py
```

In the path of the `user_relu` folder, we can create a test file that calls the custom Op we just implemented as follows:

```python
import oneflow as flow
import numpy as np
import os
import oneflow.typing as tp

# Create module objects based on the specified path and op_type_name.
module_path = os.path.dirname(os.path.abspath(__file__))
user_relu_op = flow.experimental.custom_op_module("user_relu", module_path)

# Make Op, Python API, Python Kernel enable
user_relu_op.py_api().cpp_def().py_kernel().build_load()

@flow.global_function()
def MyJob(x: tp.Numpy.Placeholder((5,), dtype=flow.float32)) -> tp.Numpy:
    with flow.scope.placement("cpu", "0:0"):
        return user_relu_op.api.user_relu_forward(x)

if __name__ == "__main__":
    input = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    output = MyJob(input)
    print(input)
    print(output)
```

In the above code, the module object is created by `flow.experimental.custom_op_module` which takes two parameters. The first one is `op_type_name` and the second one is the path to the `user_relu`. The returned `module` object represents our custom Op.

Next, the custom Op can be enabled by `user_sigmoid_op.py_api().cpp_def().py_kernel().build_load().` The Python interface of Op when it is enabled is defined in `user_relu_py_api.py' as (`user_relu_forward`). It is placed in the `api` namespace of the `moudle` object. Therefore, we need to call it in the following way:

```python
user_sigmoid_op.api.user_relu_forward(x)
```

Because the Python Kernel can only run on CPU devices which means you need to specify the calculation device as CPU:
```python
with flow.scope.placement("cpu", "0:0"):
```

## Provides Inverse Calculation for Custom Op
We have already done the forward calculation of the `user_relu` in the above steps which can be used for the job function of `type="predict"` . However, if we want to support a training job function of `type="train"`, we need to provide the reverse calculation for the custom Op as well.

The code that provides the reverse calculation for the custom Op needs to be written in the `op_type_name_cpp_def.cpp` and registered with the macro `REGISTER_USER_OP_GRAD`.

From a mathematical point of view, the registration process is where we specify a backward gradient calculation method for a custom op. From a programming point of view, it is setting up a backward-generating function for a custom Op. Within that function, writing code that specifies how the input gradient of the Op is calculated.

In the following, we will implement a special Op called `user_relu_backward`. We will use this "specially tailored" Op when registering a backward gradient for `user_relu`.

### Implementation of `user_relu_backward` Op
The process for implementing the `user_relu_backward` is almost identical to implementing the `user_relu`. First, configure and register the Op in `user_relu_cpp_def.cpp`:

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

Please pay attention that`.Attr<std::string>("device_sub_tag", "py")` must in the above code. It tells OneFlow to call the Python Kernel by default when using this Op.

By the same logic, since we don't need to call `user_relu_backward`. So you don't need to wrap the Python interface to `user_relu_py_api.py` for `user_relu_backward`.  We can implement it directly in the Python Kernel.

Implement the `backward` in `user_relu_py_kernel.py`:

```python
def backward(args):
    (y, dy) = args
    dx = (y>0)*dy
    return dx
```
Its parameter is a `tuple` which number and order are corresponds to `Input` for Op registration and the output corresponds to `Output` for Op registration.

### Register reverse gradient for Op.
We need to register backward for our forwrd Op by `REGISTER_USER_OP_GRAD` in `user_relu_cpp_def.cpp`.

For example:
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

By `REGISTER_USER_OP_GRAD("user_relu_forward")` we registers the backward gradient rule for forward Op which takes one parameter is **forward ** `op_type_name`.

Then set the backward gradient rule by `SetBackwardOpConfGenFn`. Similar to Op, register the backward in `op_type_name_cpp_def.cpp` which doesn't really do the calculation. It sets the relationship between the backward calculation and the forward then telling the OneFlow framework:

- Which Op is used to find the backward gradient
- Where does the Op input come from and how does it relate to the forward Op

In the above code:

```c++
      const auto& grad_op_func = [&ctx](user_op::BackwardOpBuilder& builder) {
        return builder.OpTypeName("user_relu_backward")
            .InputBind("y", ctx->FwOp().output("y", 0))
            .InputBind("dy", ctx->FwOp().output_grad("y", 0))
            .Output("dx")
            .Build();
      };
```

Defines a method for calculate the gradient of an Op: Use the `user_relu_backward` and take the forward output `y` as input to `user_relu_backward`. Take the gradient of the forward output `y` as input to `dy` and finally get `dx`.

We need to call the gradient calculation method by:
```cpp
ctx->DefineOp(grad_op_name, grad_op_func);
```
The following code is:
```cpp
      const auto& dx_get_func = [&ctx, &grad_op_name]() -> const std::string& {
        return ctx->GetOp(grad_op_name).output("dx", 0);
      };
      ctx->FwOp().InputGradBind(user_op::OpArg("x", 0), dx_get_func);
```

To bind the forward input `x` to the output (`dx`) of the gradient calculation method you just set. So it can be automatically derived when training with OneFlow.

## Others

- The code involved in this article can be found [here](https://github.com/Oneflow-Inc/oneflow-documentation/tree/master/cn/docs/code/extended_topics/python_op).
- More advanced settings for Op registration can be found [here](user_op.md#opregistry).
- When registering a backward gradient, it is also possible to use an existing Op without the need for a custom backward Op. More information can be found [here](./user_op.md#opgradregistry). 

