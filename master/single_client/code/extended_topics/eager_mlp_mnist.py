# eager_mlp_mnist.py
import oneflow as flow
import oneflow.typing as tp

flow.enable_eager_execution(True)
BATCH_SIZE = 100


def main(images, labels):
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
            logits = flow.layers.dense(
                hidden, 10, kernel_initializer=initializer, name="dense2"
            )
            loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

        lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
        flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)

        return loss

    return train_job(images, labels)


if __name__ == "__main__":
    check_point = flow.train.CheckPoint()
    check_point.init()

    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE, BATCH_SIZE
    )
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
        loss = main(images, labels)
        if i % 20 == 0:
            print(loss.mean())
