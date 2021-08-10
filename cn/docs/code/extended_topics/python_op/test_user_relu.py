import numpy as np
import os
from oneflow.compatible import single_client as flow
from oneflow.compatible.single_client import typing as tp

module_path = os.path.dirname(os.path.abspath(__file__))
user_relu_op = flow.experimental.custom_op_module("user_relu", module_path)
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
