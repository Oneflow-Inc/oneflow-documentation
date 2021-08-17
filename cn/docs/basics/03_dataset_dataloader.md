# Dataset 与 DataLoader

OneFlow 的 `Dataset` 与 `DataLoader` 的行为与 [PyTorch](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) 的是一致的，都是为了让数据集管理与模型训练解耦。

在 [oneflow.utils.vision.datasets](todo_refine_rst_datasets.md) 下，提供的类可以帮助我们自动下载、加载常见的数据集（如 FashionMNIST）。

`DataLoader` 将数据集封装为迭代器，方便训练时遍历并操作数据。

```python
import matplotlib.pyplot as plt

import oneflow as flow
import oneflow.nn as nn
from oneflow.utils.vision.transforms import ToTensor
from oneflow.utils.data import Dataset
import oneflow.utils.vision.datasets as datasets
```

## Dataset 加载数据

以下的例子展示了如何使用内置的 `Dataset` 加载数据。

- `root`：数据集存放的路径
- `train`： `True` 代表下载训练集、`False` 代表下载测试集
- `download=True`： 如果 `root` 路径下数据集不存在，则从网络下载
- `transforms`：指定的数据转换方式


```python
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```

第一次运行，会下载数据集，输出：

```text
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz
26422272it [00:17, 1504123.86it/s]                                                          
Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz
29696it [00:00, 98468.01it/s]                                                               
Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
4422656it [00:07, 620608.04it/s]                                                            
Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw

Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
6144it [00:00, 19231196.85it/s]                                                             
Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
```

## 遍历数据

`Dataset` 对象，可以像 `list` 一样，用下表索引，比如 `training_data[index]`。
以下的例子，随机访问 `training_data` 中的9个图片，并显示。

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

![fashionMNIST](../imgs/fashionMNIST.png)
