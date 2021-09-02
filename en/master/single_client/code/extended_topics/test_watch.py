# test_watch.py
import numpy as np
import oneflow as flow
import oneflow.typing as tp


def watch_handler(y: tp.Numpy):
    print("out:", y)


@flow.global_function()
def ReluJob(x: tp.Numpy.Placeholder((5,))) -> None:
    y = flow.nn.relu(x)
    flow.watch(y, watch_handler)


flow.config.gpu_device_num(1)
data = np.random.uniform(-1, 1, 5).astype(np.float32)
print("in:", data)
ReluJob(data)