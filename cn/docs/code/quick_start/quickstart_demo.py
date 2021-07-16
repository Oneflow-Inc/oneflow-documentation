import torch
import torch.nn as nn
import numpy as np 
import oneflow as flow

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
BATCH_SIZE = 100
(train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(BATCH_SIZE, BATCH_SIZE)

print(train_images)
tr_images = torch.LongTensor(train_images)
tr_labels = torch.LongTensor(train_labels)
te_images = torch.LongTensor(test_images)
te_labels = torch.LongTensor(test_labels)

classes = ('1', '2', '3', '4', '5', '6', '7', '8', '9', '0')

# Model
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        ) # 这里我觉得可以画一张网络图，给读者一个更清晰的视觉化的解释

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = LeNet5() # may change
loss_fn = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3) # 更新梯度

# Training Loop
num_epochs = 5
n_total_steps = 60000
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):
        # 正向和loss
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
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model")

# 预测
model.eval()
x, y = te_images[0], te_images[0]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted:"{predicted}", Actual: "{actucal}"')








