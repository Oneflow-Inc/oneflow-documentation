# DATASETS & DATALOADERS

The behavior of OneFlow's `Dataset` and `DataLoader` is the same as [PyTorch](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html). Both `Dataset` and `DataLoader` are for designed for making dataset management decoupling with model training.

[oneflow.utils.vision.datasets](https://oneflow.readthedocs.io/en/master/utils.html#module-oneflow.utils.vision.datasets) provides us a number of classes that can automatically download and load prevailing datasets (such as fashionmnist).

`DataLoader` wraps data into an iterator, for easy iterating and access to samples during training.

```python
import matplotlib.pyplot as plt

import oneflow as flow
import oneflow.nn as nn
from oneflow.utils.vision.transforms import ToTensor
from oneflow.utils.data import Dataset
import oneflow.utils.vision.datasets as datasets
```

## Loading a Dataset

Here is an example of how to load by `Dataset`.

- `root`: the path where the train/test data is stored;
- `train`: `True` for training dataset, `False` for test dataset;
- `download=True`: downloads the data from the internet if it’s not available at `root`;
- `transforms`: the feature and label transformations.


```python
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/Fashion-MNIST/",
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/Fashion-MNIST/",
)
```

The first time it runs, it will download the data set and output the following:

```text
Downloading https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/Fashion-MNIST/train-images-idx3-ubyte.gz
Downloading https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/Fashion-MNIST/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz
26422272/? [00:02<00:00, 8090800.72it/s]
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/Fashion-MNIST/train-labels-idx1-ubyte.gz
Downloading https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/Fashion-MNIST/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz
29696/? [00:00<00:00, 806948.09it/s]
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/Fashion-MNIST/t10k-images-idx3-ubyte.gz
Downloading https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/Fashion-MNIST/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
4422656/? [00:00<00:00, 19237994.98it/s]
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/Fashion-MNIST/t10k-labels-idx1-ubyte.gz
Downloading https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/mnist/Fashion-MNIST/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
6144/? [00:00<00:00, 152710.85it/s]
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```

## Iterating the Dataset

We can index `Dataset` manually like a `list`: `training_data[index]`.
The following example randomly accesses 9 pictures in `training_data` and visualizes them.

```python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
from random import randint
for i in range(1, cols * rows + 1):
    sample_idx = randint(0, len(training_data))
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze().numpy(), cmap="gray")
plt.show()
```

![fashionMNIST](./imgs/fashionMNIST.png)

## Creating a Custom Dataset for Your Files

A custom dataset can be defined by inheriting [oneflow.utils.data.Dataset](https://oneflow.readthedocs.io/en/master/utils.html?highlight=oneflow.utils.data.Dataset#oneflow.utils.data.Dataset). Custom `Dataset` can be used with `Dataloader` introduced in the next section to simplify data processing.

Here is an example of how to create a custom `Dataset`, the key steps are:

- Inheriting `oneflow.utils.data.Dataset`
- Implements the `__len__`  method that returns the number of samples in our dataset.
- Implements the `__getitem__` method that loads and returns a sample from the dataset when users call `dataset_obj[idx]`.

```python
import numpy as np
class CustomDataset(Dataset):
    raw_data_x = np.array([[1, 2], [2, 3], [4, 6], [3, 1]], dtype=np.float32)
    raw_label = np.array([[8], [13], [26], [9]], dtype=np.float32)

    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(raw_label)

    def __getitem__(self, idx):
        x = CustomDataset.raw_data_x[idx]
        label = CustomDataset.raw_label[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            label = self.target_transform(label)
        return x, label

custom_dataset = CustomDataset()
print(custom_dataset[0])
print(custom_dataset[1])
```

Output：

```text
(array([1., 2.], dtype=float32), array([8.], dtype=float32))
(array([2., 3.], dtype=float32), array([13.], dtype=float32))
```

## Using DataLoader

The Dataset retrieves all features of our dataset and labels one sample at a time. While training a model, we typically want to pass samples in "minibatches", which means they will load a same amount of data as the batch size at the time, and reshuffle the data at every epoch to reduce model overfitting.

At this time, we can use `DataLoader`. `DataLoader` can wrap `Dataset` into an iterator to access data during the training loop. Here is an example:

- `batch_size=64`: the batch size at each iteration
- `shuffle`: whether the data is shuffled after we iterate over all batches 

```python
from oneflow.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
x, label = next(iter(train_dataloader))
print(f"shape of x:{x.shape}, shape of label: {label.shape}")
```

Output：
```text
shape of x:flow.Size([64, 1, 28, 28]), shape of label: flow.Size([64])
```

```python
img = x[0].squeeze().numpy()
label = label[0]
plt.imshow(img, cmap="gray")
plt.show()
print(label)
```

Output：(output a picture randomly)

![dataloader item](./imgs/dataloader_item.png)

```text
tensor(9, dtype=oneflow.int64)
```

We can also use the `Dataloader` iterator during the training loop.

```python
for x, label in train_dataloader:
    print(x.shape, label.shape)
    # training...
```
