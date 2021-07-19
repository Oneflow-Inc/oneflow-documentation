import oneflow.experimental as flow
import oneflow.experimental.nn as nn
import numpy as np 

# 下载并设置数据
BATCH_SIZE = 100
(train_images, train_labels), (test_images, test_labels) = flow.data.load_mnist(BATCH_SIZE, BATCH_SIZE)

tr_images = flow.tensor(train_images)
tr_labels = flow.tensor(train_labels)
te_images = flow.tensor(test_images)
te_labels = flow.tensor(test_labels)

# 设置模型需要的参数
input_size = 784
hidden_size = 100
num_classes = 10
batch_size = 100

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

model = LeNet5(input_size, hidden_size1, hidden_size2, num_classes)
loss_fn = nn.CrossEntropyLoss() # loss function
optimizer = flow.optim.SGD(model.parameters(), lr=0.003) # 更新梯度

# 训练循环
num_epochs = 15
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(zip(train_images, train_labels)):

        #调整参数格式
        T_images = flow.tensor(images, dtype=flow.float32)
        T_labels = flow.tensor(labels, dtype=flow.long)
        
        #矩阵格式对齐
        T_images = flow.reshape(T_images, shape=[-1, 28*28])
        T_labels = T_labels

        #正向传播和loss
        outputs = model(T_images)
        loss = loss_fn(outputs, T_labels)

        #反向传播和更新权重
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 5000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

print('Finished Training')

# 储存模型
# flow.save(model.state_dict(), "文件夹路径")
# print("Saved OneFlow Model")

# 模型准确率
with flow.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in zip(test_images, test_labels):
        T_images = flow.tensor(images, dtype=flow.float32)
        T_labels = flow.tensor(labels, dtype=flow.float32)
        T_images = flow.reshape(T_images, shape=[-1, 28*28])
        
        T_labels = T_labels
        outputs = model(T_images)
        
        predictions = flow.argmax(outputs, dim=1)
        n_samples +=labels.shape[0]
        T_correct = flow.eq(predictions, T_labels)
        x = flow.sum(T_correct).numpy()
        n_correct += x[0]
    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc}%')

# 预测
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
model.eval()
te_images = flow.reshape(te_images, shape=[-1, 28*28])
x, y = te_images, test_labels[0]
with flow.no_grad():
    pred = model(x)
    x = pred[0].argmax().numpy()
    predicted, actual = classes[x.item(0)], y[0]
    print(f'Predicted:"{predicted}", Actual: "{actual}"')