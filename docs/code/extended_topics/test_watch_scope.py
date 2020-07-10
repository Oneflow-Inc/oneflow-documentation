import numpy as np
import oneflow as flow

def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    config.train.primary_lr(0.1)
    config.train.model_update_conf({"naive_conf": {}})
    return config

tensor_watched = {}
tensor_grad_watched = {}

@flow.global_function(get_train_config())
def train_job(images=flow.FixedTensorDef((8, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((8,), dtype=flow.int32)):
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(images, [images.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer)
    with flow.watch_scope(tensor_watched, tensor_grad_watched):
        logits = flow.layers.dense(hidden, 10, kernel_initializer=initializer)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
    flow.losses.add_loss(loss)

    return loss

flow.train.CheckPoint().init()
images = np.random.uniform(-10, 10, (8, 1, 28, 28)).astype(np.float32)
labels = np.random.randint(-10, 10, (8,)).astype(np.int32)
loss = train_job(images, labels).get().mean()

print("view watched tensors")
for lbn, tensor_data in tensor_watched.items():
    print(lbn)
    #print(tensor_data["blob"].ndarray())
    #print(tensor_data["blob_def"])

print("view watched grad tensors")
for lbn, tensor_data in tensor_grad_watched.items():
    print(lbn)
    #print(tensor_data["blob"].ndarray())
    #print(tensor_data["blob_def"])