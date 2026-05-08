

# 3维 = 时间序列
# 4维 = 图像
# 5维 = 视频

# 一个图像的表示
# (weight, height, channel) = 3D
# 机器学习中处理一批图片集合
# (batch_size, width, height, channel) = 4D

# 创建tensor
import torch
# 创建随机矩阵
x = torch.rand(4,3)
print(x)
# 创建全0矩阵
x = torch.zeros(4,3,dtype=torch.long)
print(x)
x = x.new_ones(4,3,dtype=torch.double)
print(x)
x = torch.randn_like(x, dtype=torch.float)
print(x)
print(type(x))
print(x.shape)

y = torch.rand(4,3)
print(x+y)
print(torch.add(x,y))
y.add_(x)
print(y)

# 自动求导
x = torch.ones(2,2,requires_grad=True)
print(x)

y = x ** 2
print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)

a = torch.randn(2,2)
a = ((a*3)/(a-1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

out2 = x.sum()
out2.backward()
print(x.grad)

out3 = x.sum()
x.grad.data.zero_()
out3.backward()
print(x.grad)

x = torch.randn(3, requires_grad=True)
print(x)

y = x * 2
i = 0
while y.data.norm() < 1000:
    y = y * 2
    i = i + 1
print(y)
print(i)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

print(x.requires_grad)
print((x**2).requires_grad)

with torch.no_grad():
    print((x**2).requires_grad)

x = torch.ones(1,requires_grad=True)
print(x.data)
print(x.data.requires_grad)

y = 2 * x
x.data *= 100

y.backward()
print(x)
print(x.grad)

