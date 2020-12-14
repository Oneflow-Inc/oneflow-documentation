# mlp_mnist_finetune.py
import oneflow as flow
import oneflow.typing as tp

BATCH_SIZE = 100
flow.config.enable_legacy_model_io(False)


@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
    with flow.scope.placement("cpu", "0:0"):
        initializer = flow.truncated_normal(0.1)
        reshape = flow.reshape(images, [images.shape[0], -1])
        hidden = flow.layers.dense(
            reshape,
            512,
            activation=flow.nn.relu,
            kernel_initializer=initializer,
            name="dense1",
        )
        dense2 = flow.layers.dense(
            hidden, 10, kernel_initializer=initializer, name="dense2"
        )

        dense3 = flow.layers.dense(
            dense2, 10, kernel_initializer=initializer, name="dense3"
        )
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, dense3)

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)

    return loss


if __name__ == "__main__":
    flow.load_variables(flow.checkpoint.get("./mlp_models_1"))

    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE, BATCH_SIZE
    )
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
        loss = train_job(images, labels)
        if i % 20 == 0:
            print(loss.mean())
    flow.checkpoint.save("./mlp_ext_models_1")
