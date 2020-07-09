import numpy as np
import oneflow as flow
from mnist_util import load_data

BATCH_SIZE = 100


def mlp(data):
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(data, [data.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer)
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer)


def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    config.train.primary_lr(0.1)
    config.train.model_update_conf({"naive_conf": {}})
    return config


@flow.global_function(get_train_config())
def train_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels=flow.FixedTensorDef((BATCH_SIZE,), dtype=flow.int32)):
    with flow.fixed_placement("cpu", "0:0"):
        logits = mlp(images)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
    flow.losses.add_loss(loss)
    return loss


if __name__ == '__main__':
    flow.config.enable_debug_mode(True)
    check_point = flow.train.CheckPoint()
    check_point.init()
    (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE)
    for epoch in range(1):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            loss = train_job(images, labels).get().mean()
            if i % 20 == 0: print(loss)
    check_point.save('./lenet_models_1')  # need remove the existed folder
