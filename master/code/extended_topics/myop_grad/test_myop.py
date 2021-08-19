import oneflow as flow
import numpy as np
import oneflow.typing as tp

# 加载模块
flow.config.load_library("final_myop.so")

flow.config.gpu_device_num(1)
global_storage = {}


def global_storage_setter(name):
    global global_storage

    def _set(x):
        global_storage[name] = x

    return _set


# python op wrapper function
def myop(input_blob):
    op = (
        flow.user_op_builder("myop_demo")
        .Op("myop")
        .Input("in", [input_blob])
        .Output("out")
        .Build()
    )
    return op.InferAndTryRun().SoleOutputBlob()


# 网络代码
@flow.global_function(type="train")
def MyJob(x: tp.Numpy.Placeholder((5,), dtype=flow.float32)) -> tp.Numpy:
    x += flow.get_variable(
        name="v1", shape=(5,), dtype=flow.float, initializer=flow.zeros_initializer(),
    )
    loss = myop(x)
    flow.optimizer.SGD(
        flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
    ).minimize(loss)
    flow.watch_diff(x, global_storage_setter("x1_diff"))
    return loss


@flow.global_function(type="train")
def CompareJob(x: tp.Numpy.Placeholder((5,), dtype=flow.float32)) -> tp.Numpy:
    x += flow.get_variable(
        name="v1", shape=(5,), dtype=flow.float, initializer=flow.zeros_initializer(),
    )
    loss = 3 * x * x
    flow.optimizer.SGD(
        flow.optimizer.PiecewiseConstantScheduler([], [1e-4]), momentum=0
    ).minimize(loss)
    flow.watch_diff(x, global_storage_setter("x2_diff"))
    return loss


if __name__ == "__main__":
    input = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
    output1 = MyJob(input)
    output2 = CompareJob(input)
    print("input", input)
    print("out1", output1)
    print("out2", output2)
    print("x_diff1", global_storage["x1_diff"].numpy())
    print("x_diff2", global_storage["x2_diff"].numpy())
