import numpy as np
import oneflow as flow
from mnist_util import load_data

BATCH_SIZE = 100
GPU_NUM = 2
BATCH_SIZE_PER_GPU = int(BATCH_SIZE / GPU_NUM)


def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    config.train.primary_lr(0.1)
    config.train.model_update_conf({"naive_conf": {}})
    return config


@flow.global_function(get_train_config())
def train_job(images=flow.MirroredTensorDef((BATCH_SIZE_PER_GPU, 1, 28, 28), dtype=flow.float),
              labels=flow.MirroredTensorDef((BATCH_SIZE_PER_GPU,), dtype=flow.int32)):
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(images, [images.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer)
    logits = flow.layers.dense(hidden, 10, kernel_initializer=initializer)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
    flow.losses.add_loss(loss)
    return loss


if __name__ == '__main__':
    flow.config.gpu_device_num(2)  # 设置GPU数目
    check_point = flow.train.CheckPoint()
    check_point.init()
    (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE)

    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
        images1 = images[:BATCH_SIZE_PER_GPU]
        images2 = images[BATCH_SIZE_PER_GPU:]
        labels1 = labels[:BATCH_SIZE_PER_GPU]
        labels2 = labels[BATCH_SIZE_PER_GPU:]

        imgs_list = [images1, images2]
        labels_list = [labels1, labels2]

        loss = train_job(imgs_list, labels_list).get().ndarray_list()
        total_loss = np.array([*loss[0], *loss[1]])
        if i % 20 == 0: print(total_loss.mean())