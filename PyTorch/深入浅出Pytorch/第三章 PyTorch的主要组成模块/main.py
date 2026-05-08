import torch
import torch.nn as nn
import torch.nn.functional as F

# LeNet 网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6,5)
        self.conv2 = nn.Conv2d(6, 16,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        # 方阵可以使用一个数来定义
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

# AlexNet
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1,2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1,1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1,1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1,1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        feature = self.conv(x)
        output = self.fc(feature.view(x.shape[0],-1))
        return output

net = AlexNet()
print(net)

import torch.nn as nn

# 模型的初始化
conv = nn.Conv2d(1, 3, 3)
linear = nn.Linear(10, 1)

isinstance(conv, nn.Conv2d)
isinstance(linear, nn.Conv2d)

print(conv.weight.data)
print(conv.weight.data)

torch.nn.init.kaiming_normal_(conv.weight.data)
print(conv.weight.data)
torch.nn.init.constant_(linear.weight.data, 0.3)
print(linear.weight.data)

# 初始化函数的封装
# 该段代码的流程是遍历当前模型的每一层，然后判断当前层属于什么类型， 设置不同权重值初始化方法。
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.zeros_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.3)
        elif isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight.data, 0.1)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias.data)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class MLP(nn.Module):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Conv2d(1,1,3)
        self.act = nn.ReLU()
        self.output = nn.Linear(10,1)

    def forward(self,x):
        o = self.act(self.hidden(x))
        return self.output(o)

mlp = MLP()
print(mlp.hidden.weight.data)
print("---初始化---")
mlp.apply(initialize_weights)
print(mlp.hidden.weight.data)
# 在初始化时，最好不要将模型的参数初始化为0，因为这样会导致梯度消失，从而影响模型的训练效果

# import matplotlib.pyplot as plt

# inputs = torch.linspace(-10, 10, steps=5000)
# target = torch.zeros_like(inputs)
# loss_f_smooth = nn.SmoothL1Loss(reduction='none')
# loss_smooth = loss_f_smooth(inputs, target)
# loss_f_l1 = nn.L1Loss(reduction='none')
# loss_l1 = loss_f_l1(inputs, target)
#
# plt.plot(inputs.numpy(), loss_smooth.numpy(), label="Smooth L1 Loss")
# plt.plot(inputs.numpy(), loss_l1, label="L1 loss")
# plt.xlabel('x_i - y_i')
# plt.ylabel('loss value')
# plt.legend()
# plt.grid()
# plt.show()


loss = nn.PoissonNLLLoss()
log_input = torch.randn(5,2,requires_grad=True)
target = torch.randn(5,2)
output = loss(log_input, target)
output.backward()
print("PoissonNLLoss损失函数的计算结果为", output)


inputs = torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
target = torch.tensor([[0.9, 0.05, 0.05], [0.1, 0.7, 0.2]], dtype=torch.float)
loss = nn.KLDivLoss()
output = loss(inputs,target)

print('KLDivLoss损失函数的计算结果为',output)



