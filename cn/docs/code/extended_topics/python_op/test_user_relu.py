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
    with flow.scope.placement("cpu", "0:0"):
        return user_relu_op.api.user_relu_forward(x)

if __name__ == "__main__":
    input = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    output = MyJob(input)
    print(input)
    print(output)
