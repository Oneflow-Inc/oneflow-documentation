# mlp_mnist.py
import oneflow as flow
import oneflow.typing as tp
import numpy as np

BATCH_SIZE = 100


@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
    with flow.scope.placement("cpu", "0:0"):
        reshape = flow.reshape(images, [images.shape[0], -1])
        initializer1 = flow.random_uniform_initializer(-1/28.0, 1/28.0)
        hidden = flow.layers.dense(
            reshape,
            500,
            activation=flow.nn.relu,
            kernel_initializer=initializer1,
            bias_initializer=initializer1,
            name="dense1",
        )
        initializer2 = flow.random_uniform_initializer(
            -np.sqrt(1/500.0), np.sqrt(1/500.0))
        logits = flow.layers.dense(
            hidden, 10, kernel_initializer=initializer2, bias_initializer=initializer2, name="dense2"
        )
        loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
    flow.optimizer.Adam(lr_scheduler).minimize(loss)
    return loss


if __name__ == "__main__":

    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE, BATCH_SIZE
    )
    for epoch in range(20):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            loss = train_job(images, labels)
            if i % 20 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, 20, loss.mean()))
