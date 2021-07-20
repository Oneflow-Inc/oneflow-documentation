import torch
import torch.nn as nn
import numpy as np 
import oneflow as flow
import torchvision
import torchvision.transforms as transforms

# Data manipulation
device = torch.device('cpu')
BATCH_SIZE = 100
(train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(BATCH_SIZE, BATCH_SIZE)

tr_images = torch.tensor(train_images)
tr_labels = torch.tensor(train_labels)
te_images = torch.tensor(test_images)
te_labels = torch.tensor(test_labels)

# 设置模型需要的参数
input_size = 784
hidden_size = 100
num_classes = 10
batch_size = 100

# 如果要用DataLoader: Dataloader-MNIST
# train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
#     transform=transforms.ToTensor(), download=True)
# test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
#     transform=transforms.ToTensor())

# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Model
input_size = 784
hidden_size1 = 128
hidden_size2 = 64
num_classes = 10
batch_size = 100
class LeNet5(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(LeNet5, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu1(out)
        out = self.l2(out)
        out = self.relu2(out)
        out = self.l3(out)
        return out

model = LeNet5(input_size, hidden_size1, hidden_size2, num_classes) # may change
loss_fn = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.003) # 更新梯度

# Training Loop
num_epochs = 15
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
        # 正向和loss
        T_images = torch.tensor(images, dtype=torch.float32)
        T_labels = torch.tensor(labels, dtype=torch.long)

        #矩阵格式对齐
        T_images = T_images.reshape(-1, 28*28).to(device)
        T_labels = T_labels.to(device)

        outputs = model(T_images)
        loss = loss_fn(outputs, T_labels)

        #反向和更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 5000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')

# 储存model
# torch.save(model.state_dict(), "model.pth")
# print("Saved PyTorch Model")

# 模型准确率
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in zip(test_images, test_labels):
        T_images = torch.tensor(images, dtype=torch.float32)
        T_labels = torch.tensor(labels, dtype=torch.long)
        T_images = T_images.reshape(-1, 28*28).to(device)
        T_labels = T_labels.to(device)
        outputs = model(T_images)

        _, predictions = torch.max(outputs, 1)
        n_samples +=labels.shape[0]
        n_correct = (predictions == T_labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')

# 预测
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
model.eval()
te_images = te_images.reshape(-1, 28*28).to(device)
x, y = te_images[0], test_labels[0]
with torch.no_grad():
    pred = model(x)
    print(pred)
    predicted, actual = classes[pred.argmax()], y[0]
    print(f'Predicted:"{predicted}", Actual: "{actual}"')