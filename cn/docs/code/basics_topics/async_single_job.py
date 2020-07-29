import oneflow as flow
import oneflow.typing as tp

BATCH_SIZE = 100


@flow.global_function(type="train")
def train_job(images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
              labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32)) -> tp.Callback[tp.Numpy]:
    # mlp
    initializer = flow.truncated_normal(0.1)
    reshape = flow.reshape(images, [images.shape[0], -1])
    hidden = flow.layers.dense(reshape, 512, activation=flow.nn.relu, kernel_initializer=initializer, name="hidden")
    logits = flow.layers.dense(hidden, 10, kernel_initializer=initializer, name="output")

    loss = flow.nn.sparse_softmax_cross_entropy_with_logits(labels, logits, name="softmax_loss")
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
    flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
    return loss


g_i = 0
def cb_print_loss(result: tp.Numpy):
    global g_i
    if g_i % 20 == 0:
        print(result.mean())
    g_i += 1


def main():
    check_point = flow.train.CheckPoint()
    check_point.init()

    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(BATCH_SIZE)
    for epoch in range(20):
        for i, (images, labels) in enumerate(zip(train_images, train_labels)):
            train_job(images, labels)(cb_print_loss)

    check_point.save('./mlp_models_1')  # need remove the existed folder


if __name__ == '__main__':
    main()
