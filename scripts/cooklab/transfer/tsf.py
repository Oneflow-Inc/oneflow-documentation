import oneflow as flow
from oneflow import nn
from oneflow.utils.data import DataLoader

from flowvision.models import resnet18
from flowvision.datasets import CIFAR10
import flowvision.transforms as transforms

NUM_EPOCHS = 3
BATCH_SIZE = 64
DEVICE = 'cuda' if flow.cuda.is_available() else 'cpu'

print("定义 Dataset 和 DataLoader")
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

train_dataset = CIFAR10(root='./data', train=True, transform=train_transform, download=True,
                        source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/cifar/cifar-10-python.tar.gz")
test_dataset = CIFAR10(root='./data', train=False, transform=test_transform, download=True,
                       source_url="https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/cifar/cifar-10-python.tar.gz")

train_data_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_data_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print("定义模型")
model = resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)

print("训练模型")


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
                print(
                    f'loss: {loss:>7f}  [epoch: {epoch} {batch * BATCH_SIZE:>5d}/{dataset_size:>5d}]')

        evaluate(model, test_data_loader)


def evaluate(model, data_loader):
    dataset_size = len(data_loader.dataset)
    model.eval()
    num_corrects = 0
    for images, labels in data_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        preds = model(images)
        num_corrects += flow.sum(flow.argmax(preds, dim=1) == labels)

    print('Accuracy: ', num_corrects.item() / dataset_size)


print("通过给优化器传入相应的需要优化的参数，来实现上文中提到的三种方式")

# print("第 1 种方式，对整个模型进行训练")
# optimizer = flow.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

# print("第 2 种方式，对特征提取器使用较小的学习率，对分类器使用较大的学习率")
# fc_params = list(map(id, model.fc.parameters()))
# backbone_params = filter(lambda p: id(p) not in fc_params, model.parameters())
# optimizer = flow.optim.SGD([{'params': backbone_params, 'lr': 0.0001},
#                             {'params': model.fc.parameters(), 'lr': 0.001}],
#                            momentum=0.9, weight_decay=5e-4)

print("第 3 种方式，固定特征提取器的参数，只训练分类器")
optimizer = flow.optim.SGD(model.fc.parameters(),
                           lr=0.001, momentum=0.9, weight_decay=5e-4)

print("开始训练")
loss_func = nn.CrossEntropyLoss()
train_model(model, train_data_loader, test_data_loader, loss_func, optimizer)
