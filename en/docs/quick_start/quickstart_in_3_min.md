This article introduces how to quickly get start with OneFlow. We can complete a full neural network training process just in 3 minutes.

## Example
With OneFlow installed, you can run the following command to download [mlp_mnist.py](https://github.com/Oneflow-Inc/oneflow-documentation/blob/master/en/docs/code/quick_start/mlp_mnist.py) python script from [repository](https://github.com/Oneflow-Inc/oneflow-documentation.git) and run it.

```
wget https://docs.oneflow.org/en/master/code/quick_start/mlp_mnist.py
python3 mlp_mnist.py
```

The output looks like below:
```
Epoch [1/20], Loss: 2.3155
Epoch [1/20], Loss: 0.7955
Epoch [1/20], Loss: 0.4653
Epoch [1/20], Loss: 0.2064
Epoch [1/20], Loss: 0.2683
Epoch [1/20], Loss: 0.3167
...
```

The output is a series of numbers representing the loss values while training. The goal of training is to make the loss value as small as possible. So far, you have completed a full neural network training by using OneFlow.

## Code Explanation
The following is the full code.
```python
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
```

The next section is a brief description of this code.

A special feature of OneFlow compares to other deep learning frameworks is:
```python
@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
```
`train_job` function which decorated by `@flow.global_function` is called "job function". Unless functions are decorated by `@flow.global_function`, or they can not be recognized by OneFlow.

The parameter `type` is used to specify the type of job: `type="train"` means it's a training job and `type="predict"` means evaluation or prediction job.

In OneFlow, a neural network training or prediction task needs two pieces of information:

* One part is the structure of neural network and its related parameters. These are defined in the job function which mentioned above.

* The other part is the configuration of training to the network. For example, `learning rate` and type of model optimizer. These are defined by code as below:

```python
    lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.001])
    flow.optimizer.Adam(lr_scheduler).minimize(loss)
```

Besides the job function definition and configuration which mentioned above, code in this script contains all the points of how to train a neural network.

- `flow.data.load_mnist(BATCH_SIZE,BATCH_SIZE)`: Prepare and load training data.

- ` train_job(images, labels)`: return the loss value for each iteration.

- `print(..., loss.mean())`: print loss values for every 20 iterations.

This page is just a simple example on neural network.
A more comprehensive and detailed introduction of OneFlow can be found in [Convolution Neural Network for Handwriting Recognition](lenet_mnist.md).

In addition, you can refer to [Basic topics](../basics_topics/data_input.md) to learn more about how to use OneFlow for deep learning.

Benchmarks and related scripts for some prevalent networks are also provided in repository [OneFlow-Benchmark](https://github.com/Oneflow-Inc/OneFlow-Benchmark).

## FAQ

- Getting stuck when running this script

> It may be that the incorrect proxy is set in the environment. You can cancel the proxy by first running the command
```
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
```

Then try again

- My computer can't connect to the Internet and keeps getting stuck when I run the script.

> This script will automatically download the required data file from the network. If your computer is not connected to the Internet, you will need to download it manually by clicking [here](https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist.npz) and placing it in the script `mlp_ mnist.py` in the same path and then try again.
