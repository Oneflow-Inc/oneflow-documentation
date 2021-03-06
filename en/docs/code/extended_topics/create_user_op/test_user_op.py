import oneflow as flow
import numpy as np
import oneflow.typing as tp

# Load modules
flow.config.load_library("final_relu.so")

# default configuration
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


# network code
@flow.global_function()
def MyJob(x: tp.Numpy.Placeholder((5,), dtype=flow.float32)) -> tp.Numpy:
    return myrelu(x)

if __name__ == "__main__":
    input = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    output = MyJob(input)
    print(input)
    print(output)
