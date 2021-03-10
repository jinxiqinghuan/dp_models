"""
实现LeNet来分类cifar10，10个epoch的效果为50%左右
"""
import torch
from torch import nn
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader



# 超参数设置
batch_size = 1
epoch = 10

# 数据集加载与处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0., ), (1., ))])

train_dataset = datasets.CIFAR10(root='../datasets/cifar10/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.CIFAR10(root='../datasets/cifar10/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# 设计模型

# LeNet5

class LeNet5(torch.nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 卷积（输入层， 32 * 32的图片作为输入）
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
        # 最大值池化
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        # 卷积
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
        # 最大值池化
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        # Fully connected layer
        # 将16*5*5个特征的矩阵转化为120个特征的矩阵
        self.fc1 = torch.nn.Linear(16*6*6, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool_1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool_2(x)
        x = x.view(-1, 16*6*6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.fc3(x)

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )   


class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)

        # 全连接
        self.fc1 = nn.Linear(10240, 1024)
        self.fc2 = nn.Linear(1024, 10)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        x = self.conv_last(x)

        x = x.view(-1, 10240)
        x = F.relu(self.fc1(x))
        out = F.relu(self.fc2(x)) 

        return out




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = LeNet5().to(device)
model = UNet(10).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.5)


def train(epoch):
    running_loss = 0
    for batch_idx, data in enumerate(train_loader, 0):
        # print(batch_idx)
        images, labels =data
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print("Epoch:", epoch + 1, "Batch_idx:", batch_idx + 1, "Loss:", running_loss / 300)
            # print('[%d, %5d] loss: %.3f' % (epoch + 1), batch_idx + 1, running_loss / 300)
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print("accuracy on test set : %d %% " % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()

