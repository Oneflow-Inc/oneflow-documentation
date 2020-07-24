import numpy as np
import oneflow as flow
import oneflow.typing as oft

def cb(y):
    print("out", y.ndarray())

@flow.global_function()
def ReluJob(x:oft.Numpy.Placeholder((10,))):
    y = flow.nn.relu(x)
    flow.watch(y, cb)

flow.config.gpu_device_num(1)
data = np.random.uniform(-1, 1, 10).astype(np.float32)
print("in: ", data)
ReluJob(data)
