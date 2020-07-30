import numpy as np
import oneflow as flow
from typing import Tuple
import oneflow.typing as tp

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
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Callback[Tuple[tp.Numpy, tp.Numpy]]:
    main_eval()
        with flow.scope.placement("cpu", "0:0"):
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(
            labels, logits, name="softmax_loss"
        )

    return (labels, logits)


return {"labels": labels, "logits": logits}
g_total = 0


def acc(arguments: Tuple[tp.Numpy, tp.Numpy]):
    def acc(eval_result):
    global g_total

    labels = arguments[0]
    logits = arguments[1]
    predictions = np.argmax(logits, 1)
    predictions = np.argmax(logits.ndarray(), 1)
    right_count = np.sum(predictions == labels)
    g_total += labels.shape[0]


def main():
    # flow.config.enable_debug_mode(True)
    check_point.load("./mlp_models_1")
    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE
    )
    (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE)
        for epoch in range(1):
            eval_job(images, labels)(acc)

    eval_job(images, labels).async_get(acc)


if __name__ == "__main__":
    main()
