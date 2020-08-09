This article introduces how to quickly get start with OneFlow. We can complete a full neural network training process just in 3 minutes.

## Example
With OneFlow installed, you can run the following command to download [mlp_mnist.py](https://github.com/Oneflow-Inc/oneflow-documentation/blob/master/en/docs/code/quick_start/mlp_mnist.py) python script from [repository](https://github.com/Oneflow-Inc/oneflow-documentation.git) and run it.

```shell
wget https://docs.oneflow.org/en/code/quick_start/mlp_mnist.py
python3 mlp_mnist.py
```

Output may looks like below:
```
2.7290366
0.81281316
0.50629824
0.35949975
0.35245502
...
```

The output is a series of number each representting the loss value of each round of training. The goal of training is make loss value as small as possible. So far, you have completed a full neural network training by using OneFlow.

## Code explanation
The following is the full code.
```python
# mlp_mnist.py
import oneflow as flow
import oneflow.typing as tp

BATCH_SIZE = 100


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


if __name__ == "__main__":
    check_point = flow.train.CheckPoint()
    check_point.init()

    (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(
        BATCH_SIZE, BATCH_SIZE
    )
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
        loss = train_job(images, labels)
        if i % 20 == 0:
            print(loss.mean())
```

The next section is a brief description of this code.

The special feature of OneFlow compare to other deep learning framework is:
```python
@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
```
`Train_job`function which decorated by `@flow.global_function` is called "job function". Only function decorated by `@flow.global_function` can be recognized by OneFlow. 

The parameter `type` is used to specify the type of job: `type="train"` means it's a training job and `type="predict"` means evaluation or prediction job.

In OneFlow, a neural network training or prediction task need two pieces of information:

* One part is structure of neural networks and related parameters. These is defined in the job function which mentioned above.

* Another part is the configuration of training to the network. For example, `learning rate` and type of model optimizer. These defined by code like below:

```
lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])
flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)
```

Besides the job function defined and related configuration which mentioned above, code in this script contains all the points of how to training a neural network.

* `check_point.init()`: Model initialization;

* `load_data(BATCH_SIZE)`: Data loading;

* `loss = train_job(images, labels)`: Return the loss value of each iteration;

* `if i % 20 == 0: print(loss)`: Print a loss value once every 20 iteration;

This page is a just a simple example on neural network. 
A more comprehensive and detailed introduction of OneFlow can be found in [Convolution Neural Network for Handwriting Recognition](lenet_mnist.md). 

In addition, you can reference to [Based topics](../basics_topics/data_input.md) to learn more about how to use OneFlow for deep learning.

Benchmarks and related scripts for some prevalent network are also provided in repository [OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark).




