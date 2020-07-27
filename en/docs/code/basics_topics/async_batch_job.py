import numpy as np
import oneflow as flow
from typing import Tuple
import oneflow.typing as oft

BATCH_SIZE = 100


def mlp(data):
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(data, [data.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name="hidden")
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="output")


def get_eval_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config


@flow.global_function(get_eval_config())
def eval_job(images:oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
             labels:oft.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)) -> oft.Callback[Tuple[oft.Numpy, oft.Numpy]]:
    main_eval()
        with flow.scope.placement("cpu", "0:0"):
        logits = mlp(images)

    return (labels, logits)


return {"labels": labels, "logits": logits}
g_total = 0
def acc(arguments:Tuple[oft.Numpy, oft.Numpy]):
    def acc(eval_result):
    global g_total

    labels = arguments[0]
    logits = arguments[1]
    predictions = np.argmax(logits, 1)
    predictions = np.argmax(logits.ndarray(), 1)
    right_count = np.sum(predictions == labels)
    g_total += labels.shape[0]


g_correct += right_count
    # flow.config.enable_debug_mode(True)
    check_point = flow.train.CheckPoint()
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(BATCH_SIZE)
    (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE)
        for epoch in range(1):
            eval_job(images, labels)(acc)

    eval_job(images, labels).async_get(acc)

print("accuracy: {0:.1f}%".format(g_correct * 100 / g_total))
    if __name__ == '__main__':
