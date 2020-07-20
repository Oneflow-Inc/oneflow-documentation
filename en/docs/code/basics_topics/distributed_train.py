import oneflow as flow

BATCH_SIZE = 100
DATA_DIRECTORY = '/dataset/mnist_kaggle/60/train'
IMG_SIZE = 28
NUM_CHANNELS = 1


def _data_load_layer(data_dir=DATA_DIRECTORY, arg_data_part_num=1, fromat="NHWC"):
    if fromat == "NHWC":
        image_shape = (IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
    else:
        image_shape = (NUM_CHANNELS, IMG_SIZE, IMG_SIZE)
    image_blob_conf = flow.data.BlobConf("img_raw", shape=image_shape,
                                         dtype=flow.float32, codec=flow.data.RawCodec())
    label_blob_conf = flow.data.BlobConf("label", shape=(1, 1), dtype=flow.int32,
                                         codec=flow.data.RawCodec())
    return flow.data.decode_ofrecord(data_dir, (label_blob_conf, image_blob_conf),
                                     data_part_num=arg_data_part_num, name="decode", batch_size=BATCH_SIZE)


def lenet(data, train=False):
    initializer = flow.truncated_normal(0.1)
    conv1 = flow.layers.conv2d(data, 32, 5, padding='SAME', activation=flow.nn.relu,
                               kernel_initializer=initializer)
    pool1 = flow.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME')
    conv2 = flow.layers.conv2d(pool1, 64, 5, padding='SAME', activation=flow.nn.relu,
                               kernel_initializer=initializer)
    pool2 = flow.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME')
    reshape = flow.reshape(pool2, [pool2.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer)
    if train: hidden = flow.nn.dropout(hidden, rate=0.5)
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer)


def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    config.train.primary_lr(0.1)
    config.train.model_update_conf({"naive_conf": {}})
    return config


@flow.global_function(get_train_config())
def train_job():
    (labels, images) = _data_load_layer(arg_data_part_num=60)

    logits = lenet(images, train=True)
    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
    flow.losses.add_loss(loss)
    return loss


def config_distributed():
    # 每个节点的gpu使用数目
    flow.config.gpu_device_num(1)
    # 通信端口
    flow.env.ctrl_port(9988)

    # 节点配置
    nodes = [{"addr": "192.168.1.12"}, {"addr": "192.168.1.11"}]
    flow.env.machine(nodes)


def main():
    config_distributed()
    check_point = flow.train.CheckPoint()
    check_point.init()

    for step in range(50):
        losses = train_job().get()
        print("{:12} {:>12.10f}".format(step, losses.mean()))

    check_point.save('./lenet_models_1')  # need remove the existed folder
    print("model saved")


if __name__ == '__main__':
    main()
