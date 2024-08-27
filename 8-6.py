import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch.nn.functional as F

# 获取 SimHei 字体的路径
font_paths = [f.fname for f in fm.fontManager.ttflist if 'SimHei' in f.name]
simhei_path = font_paths[0] if font_paths else None

if not simhei_path:
    raise ValueError("SimHei 字体未找到，请确保已安装 SimHei 字体。")

my_font = fm.FontProperties(fname=simhei_path)

# 检测CUDA可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载数据集
train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# 构建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 6 * 6, 512)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool2(x)
        x = self.dropout2(x)
        x = x.view(-1, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# 训练模型
num_epochs = 20
train_losses, val_losses, train_accs, val_accs = [], [], [], []
for epoch in range(num_epochs):
    train_loss = 0.0
    train_correct = 0
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output, 1)
        train_correct += (predicted == target).sum().item()
    model.eval()
    val_loss = 0.0
    val_correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output, 1)
            val_correct += (predicted == target).sum().item()

    train_loss /= len(train_data)
    val_loss /= len(test_data)
    train_acc = train_correct / len(train_data)
    val_acc = val_correct / len(test_data)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    print(
        f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

# 绘制损失函数和准确率曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 绘制损失函数子图
ax1.plot(train_losses, label='训练损失', color='blue')
ax1.plot(val_losses, label='验证损失', color='orange')
ax1.set_xlabel('迭代次数', fontproperties=my_font)
ax1.set_ylabel('损失', fontproperties=my_font)
ax1.legend(prop=my_font)
ax1.set_title('损失函数曲线', fontproperties=my_font)

# 绘制准确率子图
ax2.plot(train_accs, label='训练准确率', color='blue')
ax2.plot(val_accs, label='验证准确率', color='orange')
ax2.set_xlabel('迭代次数', fontproperties=my_font)
ax2.set_ylabel('准确率', fontproperties=my_font)
ax2.legend(prop=my_font)
ax2.set_title('准确率曲线', fontproperties=my_font)

plt.show()
