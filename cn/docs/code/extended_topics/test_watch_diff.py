import numpy as np
import oneflow as flow
import oneflow.typing as oft

def get_cb(bn):
    def cb(x):
        blob = x.ndarray()
        print(bn, blob.shape, blob.dtype)
        #print(blob)
    return cb

def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    config.train.primary_lr(0.1)
    config.train.model_update_conf({"naive_conf": {}})
    return config

@flow.global_function(get_train_config())
def train_job(images:oft.Numpy.Placeholder((8, 1, 28, 28), dtype=flow.float),
              labels:oft.Numpy.Placeholder((8,), dtype=flow.int32)):
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(images, [images.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer)
    logits = flow.layers.dense(hidden, 10, kernel_initializer=initializer)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
    flow.losses.add_loss(loss)

    flow.watch(logits, get_cb("logits"))
    flow.watch_diff(logits, get_cb("logits_grad"))

    return loss

flow.train.CheckPoint().init()
images = np.random.uniform(-10, 10, (8, 1, 28, 28)).astype(np.float32)
labels = np.random.randint(-10, 10, (8,)).astype(np.int32)
loss = train_job(images, labels).get().mean()
print(loss)
