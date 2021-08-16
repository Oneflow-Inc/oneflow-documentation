"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os

import oneflow as flow
import oneflow.nn as nn
import oneflow.utils.vision.transforms as transforms


batch_size=128

# Dataloader 设置
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
train_iter = flow.utils.data.DataLoader(
    mnist_train, batch_size, shuffle=True
)
test_iter = flow.utils.data.DataLoader(
    mnist_test, batch_size, shuffle=False
)

# 设置模型需要的参数
input_size = 784
hidden_size1 = 128
num_classes = 10


# 具体模型
class Net(nn.Module):
    def __init__(self, input_size, hidden_size1, num_classes):
        super(Net, self).__init__()
        self.flatten = nn.Flatten()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size1, hidden_size1)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size1, num_classes)
    def forward(self, x):
        x = self.flatten(x)
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        return out

device = flow.device("cuda")
model = Net(input_size, hidden_size1, num_classes)
print(model)
model.to(device)

loss = nn.CrossEntropyLoss().to(device)
optimizer = flow.optim.SGD(model.parameters(), lr=0.003)

# 精度测试
def evaluate_accuracy(data_iter, net, device=None):
    n_correct, n_samples = 0.0, 0
    net.to(device)
    net.eval()
    with flow.no_grad():
        for images, labels in data_iter:
            images = images.reshape((-1, 28*28))
            images = images.to(device=device)
            labels = labels.to(device=device)
            n_correct += (net(images).argmax(dim=1).numpy() == labels.numpy()).sum()
            n_samples += images.shape[0]
    net.train()
    return n_correct/n_samples

# 训练模型
def train(dataloader, model, loss_fn, optimizer):
    train_loss, n_correct, n_samples = 0.0, 0.0, 0
    for images, labels in train_iter:
        images = images.reshape((-1, 28*28))
        images = images.to(device=device)
        labels = labels.to(device=device)
        features = model(images)
        l = loss(features, labels).sum()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        train_loss += l.numpy()
        n_correct += (features.argmax(dim=1).numpy() == labels.numpy()).sum()
        n_samples += images.shape[0]
    

# 具体训练循环
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, n_correct, n_samples = 0.0, 0.0, 0
    train(train_iter, model, loss, optimizer)
    evaluate_accuracy(test_iter, model, device)
     # 验证精度
    test_acc = evaluate_accuracy(test_iter, model, device)

    print("epoch %d, test acc %.3f" % 
        ( epoch + 1, test_acc))
print("Done!")

# 储存模型
flow.save(model.state_dict(), "./mnist_model")
print("Saved OneFlow Model")

test_net = Net(input_size, hidden_size1, num_classes)
test_net.load_state_dict(flow.load("./mnist_model"))

# 预测
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
test_net.eval()
for images, labels in test_iter:
    pred = test_net(images.reshape((-1, 28*28)))
    x = pred[0].argmax().numpy()
    predicted, actual = classes[x.item(0)], labels[0].numpy()
    print(f'Predicted:"{predicted}", Actual: "{actual}"')
    break
