import numpy as np
import oneflow as flow
from typing import Tuple
import oneflow.typing as oft

BATCH_SIZE = 100


def mlp(data):
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(data, [data.shape[0], -1])
    hidden = flow.layers.dense(
        reshape,
        512,
        activation=flow.nn.relu,
        kernel_initializer=initializer,
        name="hidden",
    )
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="output")


@flow.global_function(type="predict")
def eval_job(
    images: oft.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: oft.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> oft.Callback[Tuple[oft.Numpy, oft.Numpy]]:
    with flow.scope.placement("cpu", "0:0"):
        logits = mlp(images)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, logits, name="softmax_loss"
        )

    return (labels, logits)


g_total = 0
g_correct = 0


def acc(arguments: Tuple[oft.Numpy, oft.Numpy]):
    global g_total
    global g_correct

    labels = arguments[0]
    logits = arguments[1]
    predictions = np.argmax(logits, 1)
    right_count = np.sum(predictions == labels)
    g_total += labels.shape[0]
    g_correct += right_count


def main():
    check_point = flow.train.CheckPoint()
    check_point.load("./mlp_models_1")
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE
    )
    for epoch in range(1):
        for i, (images, labels) in enumerate(zip(test_images, test_labels)):
            eval_job(images, labels)(acc)

    print("accuracy: {0:.1f}%".format(g_correct * 100 / g_total))


if __name__ == "__main__":
    main()
