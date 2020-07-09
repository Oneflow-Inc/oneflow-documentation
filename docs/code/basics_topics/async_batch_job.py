import numpy as np
import oneflow as flow
from mnist_util import load_data

BATCH_SIZE = 100


def mlp(data):
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(data, [data.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer)
    return flow.layers.dense(hidden, 10, kernel_initializer=initializer)


def get_eval_config():
    config = flow.function_config()
    config.default_data_type(flow.float)
    return config


@flow.global_function(get_eval_config())
def eval_job(images=flow.FixedTensorDef((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
             labels=flow.FixedTensorDef((BATCH_SIZE,), dtype=flow.int32)):
    with flow.fixed_placement("cpu", "0:0"):
        logits = mlp(images)
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")

    return {"labels": labels, "logits": logits}


g_total = 0
g_correct = 0


def acc(eval_result):
    global g_total
    global g_correct

    labels = eval_result["labels"]
    logits = eval_result["logits"]

    predictions = np.argmax(logits.ndarray(), 1)
    right_count = np.sum(predictions == labels)
    g_total += labels.shape[0]
    g_correct += right_count


def main_eval():
    # flow.config.enable_debug_mode(True)
    check_point = flow.train.CheckPoint()
    check_point.load('./mlp_models_1')
    (train_images, train_labels), (test_images, test_labels) = load_data(BATCH_SIZE)
    for epoch in range(1):
        for i, (images, labels) in enumerate(zip(test_images, test_labels)):
            eval_job(images, labels).async_get(acc)

    print("accuracy: {0:.1f}%".format(g_correct * 100 / g_total))


if __name__ == '__main__':
    main_eval()