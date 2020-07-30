# test_watch_diff.py
import oneflow as flow
import oneflow.typing as tp

BATCH_SIZE = 100

def watch_diff_handler(blob: tp.Numpy):
    print("watch_diff_handler:", blob, blob.shape, blob.dtype)

def get_train_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config


@flow.global_function(type="train", function_config=get_train_config())
def train_job(images:tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels:tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)) -> tp.Numpy:
    with flow.scope.placement("cpu", "0:0"):
        initializer = flow.truncated_normal(0.1)
        reshape = flow.reshape(images, [images.shape[0], -1])
        hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name="hidden")
        logits = flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="output")
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
    
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    flow.watch_diff(logits, watch_diff_handler)
    return loss


if __name__ == '__main__':
    check_point = flow.train.CheckPoint()
    check_point.init()

    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(BATCH_SIZE)
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
        loss = train_job(images, labels)
        if i % 20 == 0: print(loss.mean())
