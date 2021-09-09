# Quickstart

This section will take the training process of MNIST as an example to briefly show how OneFlow can be used to accomplish common tasks in deep learning. Refer to the links in each section to the presentation on each subtask.

Let’s start by importing the necessary libraries:

```python
import oneflow as flow
import oneflow.nn as nn
import oneflow.utils.vision.transforms as transforms

BATCH_SIZE=128
```


## Working with Data

OneFlow has two primitives to work with data, which are Dataset and Dataloader.

The [oneflow.utils.vision.datasets](https://oneflow.readthedocs.io/en/master/utils.html#module-oneflow.utils.vision.datasets) module contains a number of real data sets (such as MNIST, CIFAR 10, FashionMNIST).

We can use `oneflow.utils.vision.datasets.MNIST` to get the training set and test set data of MNIST.

```python
mnist_train = flow.utils.vision.datasets.MNIST(
    root="data",
    train=True,
    transform=transforms.ToTensor(),
    download=True,
    source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/MNIST/",
)
mnist_test = flow.utils.vision.datasets.MNIST(
    root="data",
    train=False,
    transform=transforms.ToTensor(),
    download=True,
    source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/MNIST/",
)
```
Out:

```text
Downloading https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/MNIST/train-images-idx3-ubyte.gz
Downloading https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/MNIST/train-images-idx3-ubyte.gz to data/MNIST/raw/train-images-idx3-ubyte.gz
9913344it [00:00, 36066177.85it/s]
Extracting data/MNIST/raw/train-images-idx3-ubyte.gz to data/MNIST/raw
...
```

This will download and extract the data set to.`./data` directory.

A [oneflow.utils.data.DataLoader](https://oneflow.readthedocs.io/en/master/utils.html#oneflow.utils.data.DataLoader) wraps an iterable around the `dataset`.


```python
train_iter = flow.utils.data.DataLoader(
    mnist_train, BATCH_SIZE, shuffle=True
)
test_iter = flow.utils.data.DataLoader(
    mnist_test, BATCH_SIZE, shuffle=False
)

for x, y in train_iter:
    print("x.shape:", x.shape)
    print("y.shape:", y.shape)
    break
```

Out:

```text
x.shape: flow.Size([128, 1, 28, 28])
y.shape: flow.Size([128])
```

> [:link: Dataset and Dataloader](./03_dataset_dataloader.md){ .md-button .md-button--primary}


## Building Network

To define a neural network in OneFlow, we create a class that inherits from `nn.Module`. We define the layers of the network in the `__init__` function and specify how data will pass through the network in the `forward` function. 


```python
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
print(model)
```

Out:

```text
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
    (5): ReLU()
  )
)
```

> [:link: Build Network](./04_build_network.md){ .md-button .md-button--primary}


## Training Model

To train a model, we need a loss function (`loss_fn`) and an optimizer (`optimizer`).The loss function is used to evaluate the difference between the prediction of the neural network and the real label. The optimizer adjusts the parameters of the neural network to make the prediction more and more close to the real label (standard answer). Here, we use [oneflow.optim.SGD](https://oneflow.readthedocs.io/en/master/optim.html?highlight=optim.SGD#oneflow.optim.SGD) to be our optimizer. This process is called back propagation.

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = flow.optim.SGD(model.parameters(), lr=1e-3)
```

The `train` function is defined for training. In a single training loop, the model make forward propagation, calculating loss, and back propagation to update model parameters.


```python
def train(iter, model, loss_fn, optimizer):
    size = len(iter.dataset)
    for batch, (x, y) in enumerate(iter):
        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current = batch * BATCH_SIZE
        if batch % 100 == 0:
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```

We also define a `test` function to verify the accuracy of the model:


```python
def test(iter, model, loss_fn):
    size = len(iter.dataset)
    num_batches = len(iter)
    model.eval()
    test_loss, correct = 0, 0
    with flow.no_grad():
        for x, y in iter:
            pred = model(x)
            test_loss += loss_fn(pred, y)
            bool_value = (pred.argmax(1).to(dtype=flow.int64)==y)
            correct += float(bool_value.sum().numpy())
    test_loss /= num_batches
    print("test_loss", test_loss, "num_batches ", num_batches)
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}, Avg loss: {test_loss:>8f}")
```

Now we can use the `train` function to begin the train process for several epochs and then use the `test` function to assess the accuracy of the network at the end of each epoch:


```python
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_iter, model, loss_fn, optimizer)
    test(test_iter, model, loss_fn)
print("Done!")
```

Out:

```text
loss: 2.299633
loss: 2.303208
loss: 2.298017
loss: 2.297773
loss: 2.294673
loss: 2.295637
Test Error:
 Accuracy: 22.1%, Avg loss: 2.292105

Epoch 2
-------------------------------
loss: 2.288640
loss: 2.286367
...
```

> [:link: Autograd](./05_autograd.md){ .md-button .md-button--primary}
> [:link: Backpropagation and Optimizer](./06_optimization.md){ .md-button .md-button--primary}

## Saving and Loading Model

Use [oneflow.save](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.save#oneflow.save) to save the model. The saved model can be then loaded by [oneflow.load](https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=oneflow.load#oneflow.load) to make predictions on it.

```python
flow.save(model.state_dict(), "./model")
```
> [:link: Model Load and Save](./07_model_load_save.md){ .md-button .md-button--primary}
## QQ Group
Any problems encountered during the install or usage, welcome to join the QQ Group to discuss with OneFlow developer and enthusiasts:

Add QQ Group by 331883 or SCAN QR code below:

![OneFlow QQ Group](./imgs/qq_group.png)
