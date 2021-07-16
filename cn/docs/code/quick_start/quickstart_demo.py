import torch
import torch.nn as nn
import numpy as np 
import oneflow as flow
import torchvision
import torchvision.transforms as transforms

# # Download training data from open datasets.
# training_data = datasets.FashionMNIST(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor(),
# )

# # Download test data from open datasets.
# test_data = datasets.FashionMNIST(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor(),
# )

# Data manipulation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# BATCH_SIZE = 100
# (train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(BATCH_SIZE, BATCH_SIZE)


# print(train_labels.shape)
# tr_images = torch.tensor(train_images)
# tr_labels = torch.tensor(train_labels)
# te_images = torch.tensor(test_images)
# te_labels = torch.tensor(test_labels)
input_size = 784
hidden_size = 100
num_classes = 10
batch_size = 100

# Dataloader-MNIST
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
    transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0')

# Model
input_size = 784
hidden_size = 100
num_classes = 10
batch_size = 100
class LeNet5(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LeNet5, self).__init__()
        # self.flatten = nn.Flatten()
        # self.linear_relu_stack = nn.Sequential(
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
            # nn.ReLU(),
            # nn.Linear(512, 10),
            # nn.ReLU()
        # ) 

    def forward(self, x):
        # x = self.flatten(x)
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out

model = LeNet5(input_size, hidden_size, num_classes) # may change
loss_fn = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # 更新梯度

# Training Loop
num_epochs = 5
n_total_steps = 60000
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 正向和loss
        # T_images = torch.tensor(images)
        # T_labels = torch.tensor(labels)


        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

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

# 预测
# model.eval()
# x, y = te_images[0], te_images[0]
# with torch.no_grad():
#     pred = model(x)
#     predicted, actual = classes[pred[0].argmax(0)], classes[y]
#     print(f'Predicted:"{predicted}", Actual: "{actucal}"')

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predictions = torch.max(outputs, 1)
        n_samples +=labels.shape[0]
        n_correct = (predictions == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}')








