# Transfer Learning in Computer Vision

This tutorial introduces the fundamentals of transfer learning and shows an example of the use of transfer learning in the field of computer vision.

## Introduction of Fundamentals

**Transfer Learning**, is a method of transferring knowledge learned from a source dataset to a target dataset.

As we all know, supervised learning is a fairly common training method for deep learning models, but it requires a large amount of labeled data to achieve good results. When we want to apply a model to a specific task, we usually cannot obtain a large amount of labeled data due to the cost. If we directly train on such small-scale data, it is easy to cause overfitting. Therefore, transfer learning is one of the ways to solve this problem.

For example, in the common image classification task in the field of computer vision, the general image classification model can be divided into two parts: feature extractor (or called backbone network) and classifier (or called output layer). Feature extractors are generally multi-layer networks such as Convolutional Neural Networks, and classifiers are generally single-layer networks such as fully connected layers. Since the categories of different classification tasks are generally different, classifiers are usually not reusable, while feature extractors are usually reusable. Although objects in the source dataset may be very different, or even have no connection at all with the target dataset, models pretrained on large-scale data may have the ability to extract more general image features (such as edges, shapes, and textures), which can help effectively identify objects in the target dataset.

Suppose we already have a pretrained model, which can be used in roughly three ways.

1. **Initialize the feature extractor with the parameters of the pretrained model, then train the entire model.** For deep learning models, the method of parameter initialization is very important to maintain numerical stability. Improper initialization methods may lead to the problem of gradient explosion or gradient disappearance during training. If a pretrained model is used for initialization, the rationality of the initial values of the model parameters can be guaranteed to a large extent, allowing the model to get a head start.

2. **Train the entire model, but use a smaller learning rate for the feature extractor and a larger learning rate for the classifier.** The pretrained feature extractor has been fully trained, so only a small learning rate is required; while the parameters of the classifier are usually initialized randomly, and it needs to be learned from scratch, so a large learning rate is required.

3. **Fix the parameters of the feature extractor and only train the classifier.** This is generally efficient and fast if the categories of the target dataset happen to be a subset of the source dataset.


## The Example of Transfer Learning

In this section, ResNet-18 is used as the feature extractor for image classification task on [CIFAR-10 Dataset](http://www.cs.toronto.edu/~kriz/cifar.html)

Pretrained models for ResNet-18 (trained on ImageNet dataset), and CIFAR-10 dataset are both conveniently available through [FlowVision](https://github.com/Oneflow-Inc/vision).

First import the required dependencies:

```python
import oneflow as flow
from oneflow import nn
from oneflow.utils.data import DataLoader

from flowvision.models import resnet18
from flowvision.datasets import CIFAR10
import flowvision.transforms as transforms
```

Define epoch, batch size, and the computing device used：
```python
NUM_EPOCHS = 3
BATCH_SIZE = 64
DEVICE = 'cuda' if flow.cuda.is_available() else 'cpu'
```

### Data Loading and Preprocessing

Define Dataset and DataLoader:

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = CIFAR10(root='./data', train=True, transform=train_transform, download=True, source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/cifar/cifar-10-python.tar.gz")
test_dataset = CIFAR10(root='./data', train=False, transform=test_transform, download=True, source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/cifar/cifar-10-python.tar.gz")

train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
```

### Define the Model

```python
model = resnet18(pretrained=True)
```
Here, we get the ResNet-18 model loaded with pretrained weights by setting the `pretrained` parameter to `True`. If we output `model.fc`, we will get "Linear(in_features=512, out_features=1000, bias=True)". We can see that this classifier has 1000 output neurons, corresponding to 1000 categories of ImageNet. The CIFAR-10 dataset has 10 classes, so we need to replace this fully connected layer classifier:

```python
model.fc = nn.Linear(model.fc.in_features, 10)
```


### Train the Model

Define the training function:
```python
def train_model(model, train_data_loader, test_data_loader, loss_func, optimizer):
    dataset_size = len(train_data_loader.dataset)
    model.train()
    for epoch in range(NUM_EPOCHS):
        for batch, (images, labels) in enumerate(train_data_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            preds = model(images)
            loss = loss_func(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                print(f'loss: {loss:>7f}  [epoch: {epoch} {batch * BATCH_SIZE:>5d}/{dataset_size:>5d}]')
    
        evaluate(model, test_data_loader)
```

Define the evaluation function with the accuracy rate as the evaluation metric:
```python
def evaluate(model, data_loader):
    dataset_size = len(data_loader.dataset)
    model.eval()
    num_corrects = 0
    for images, labels in data_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        preds = model(images)
        num_corrects += flow.sum(flow.argmax(preds, dim=1) == labels)

    print('Accuracy: ', num_corrects.item() / dataset_size)
```
The three methods mentioned above can be implemented by passing in the corresponding parameters that need to be optimized to the optimizer.

Method 1: Train the entire model

```python
optimizer = flow.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
```
Method 2: Use a smaller learning rate for the feature extractor and a larger learning rate for the classifier

```python
fc_params = list(map(id, model.fc.parameters()))
backbone_params = filter(lambda p: id(p) not in fc_params, model.parameters())
optimizer = flow.optim.SGD([{'params': backbone_params, 'lr': 0.0001},
                            {'params': model.fc.parameters(), 'lr': 0.001}],
                            momentum=0.9, weight_decay=5e-4)
```
Method 3: Fix the parameters of the feature extractor and only train the classifier

```python
optimizer = flow.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
```

Start training:

```python
loss_func = nn.CrossEntropyLoss()
train_model(model, train_data_loader, test_data_loader, loss_func, optimizer)
```

### Comparison of Results

In the case of using transfer learning (the first method is used here), the accuracy of the model on the test set after training for 3 epochs reaches **0.9017**; If we train from scratch without transfer learning, the accuracy is only **0.4957** after the same 3 epochs of training. This shows that transfer learning can indeed play a significant role.
