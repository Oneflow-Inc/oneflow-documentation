This article introduces how to quickly get starte with OneFlow. We can complete a full neural network training process just in 3 minutes.

## Example
If you already have one flow installed, you can run the following command to clone our [repository](https://github.com/Oneflow-Inc/oneflow-documentation.git) and run the script called [mlp_mnist.py](https://github.com/Oneflow-Inc/oneflow-documentation/blob/master/docs/code/quick_start/mlp_mnist.py) and run.

```shell
wget https://docs.oneflow.org/code/quick_start/mlp_mnist.py 

```

Then run the neural network training script:
```shell
python mlp_mnist.py
```

We will get following output:
```
2.7290366
0.81281316
0.50629824
0.35949975
0.35245502
...
```

The output is a string of number, each number represents the loss value of each round of training.The target of training is make loss value as small as possible. Thus, you have completed a full neural network training by using OneFlow.

## Script interpretation
The following is the full script
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

The next chapter is a brief description of this script.

The special feature of OneFlow compare to other deep learning framework:
```python
@flow.global_function(type="train")
def train_job(
    images: tp.Numpy.Placeholder((BATCH_SIZE, 1, 28, 28), dtype=flow.float),
    labels: tp.Numpy.Placeholder((BATCH_SIZE,), dtype=flow.int32),
) -> tp.Numpy:
```
`Train_job`function which decorated by `@flow.global_function`is called "job function". Only been decorated by `@flow.global_function` can be identified by OneFlow. Use "type" to specified job type: type="train" means training job and type="predict" means evaluation or prediction job.

In OneFlow, a neural network training or prediction task need two pieces of information:

* One part is structure related parameters of the neural network itself. These is defined in the job function which mentioned above.

* Another part is using what kind of configuration to train the network. For example, `learning rate` and method of update model optimization. The configuration of `get_train_config()` in `@flow.global_function(get_train_config())` like below:

`lr_scheduler = flow.optimizer.PiecewiseConstantScheduler([], [0.1])`
  `flow.optimizer.SGD(lr_scheduler, momentum=0).minimize(loss)`

This part of script contains all the elements of training a neural network besides the job function and its configuration which mentioned above.

- `check_point.init()`: Initialize the network model parameters;

- `load_data(BATCH_SIZE)`: Prepare and load training data;

- `train_job(images, labels).get().mean()`: Return the loss value of each training;

- `if i % 20 == 0: print(loss)`: Print a loss value for each 20 times of training;




This page is a demonstration of a simple network. In Using [convolution neural network for handwriting recognition](lenet_mnist.md), we will give a more comprehensive and detailed introduction of using OneFlow process. In addition, you can reference to the training of all kinds of problems in detail in [Based topics](../basics_topics/data_input.md) of OneFlow.


We also provide some of the prevalent network and the data for you to reference [sample code](https://github.com/Oneflow-Inc/OneFlow-Benchmark).




