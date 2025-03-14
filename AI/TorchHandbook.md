# PyTorch 简介
> PyTorch 是一个灵活且高效的深度学习框架，提供动态图计算、自动求导 和 GPU 加速，广泛应用于研究和生产，其书张量计算API与numpy类似。
# PyTorch 白手入门
## 1. 判断cuda是否可用
> ```Python
> import torch
> print(torch.cuda.is_avaliable())
>
> # 张量迁移至GPU，安全的方式
> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
> a = torch.IntTensor([2,3,4])
> a = a.to(device)
> b = torch.IntTensor([3,4,5])
> b = b.to(device)
> m = a*b
> # 如果要将张量从显存->内存,转换成numpy打印
> print(m.cpu().numpy())
> ```
> ---
## 2. 简单线性回归
### 2.1 导入必要的库
> ```Python
> import torch
> from torch import optim
> ```
> * torch: Pytorch的核心库，提供对张量(Tensor)运算的支持。
> * optim: PyTorch中的优化器库，提供了各种优化算法(如梯度下降、Adam等)。
### 2.2 构建模型
> ```Python
> def build_model():
>     model = torch.nn.Sequential()
>     model.add_module("linear",torch.nn.Linear(1,1,bias=False))
> ```
> * torch.nn.Sequential(): 一个容器，按顺序添加多个层，构建神经网络模型。在这个例子中，我们只用一个线性层。
> * torch.nn.Linear(1,1,bias=False): 这是一个全连接层(线性层)，输入和输出的维度都是1。bias=False表示的是没有偏置项(通常在线性回归中，偏置项是可选的)。该层的的作用是计算线性变换 y=wx+b 偏置项b被禁用了
### 2.3 训练函数
> ```Python
> def train(model,loss,optimizer,x,y):
>     model.train()
>     x = x.requires_grad_(False)  # 原地修改
>     y = y.requires_grad_(False)  # 原地修改
>     optimizer.zero_grad()  # Reset gradient
>     fx = model.forward(x.view(len(x),1)).squeeze()  # Forward
>     output = loss.forward(fx,y)
>     output.backward()  # Bachward
>     optimizer.step()  # Update parameters
>     return output.item()
> ```
> 这个函数是**训练模型**的关键步骤,包含了**前向传播**、**损失计算**、**反向传播**和**参数更新**：
> * model.train(): 将模型设置为训练模式。
> * x.requires_grad_(False)与y.requires_grad_(False)：显示声明两个张量不需要计算梯度。输入数据和标签通常不需要梯度，只需要进行前向传播计算。
> * optimizer.zero_grad(): 清空之前的梯度，防止累积(PyTorch默认是累积梯度)。
> * model.forward(x.view(len(x),1)): 进行前向传播。x.view(len(x),1)将输入x重塑成一个形状为(len(x),1)的张量。通过model.forward()得到模型的输出。
> * loss.forward(fx,y): 计算预测结果fx和真实标签y之间的损失，这里使用的是**均方误差损失(MES Loss)**。
> * output.backward(): 进行反向传播，计算每个参数的梯度。
> * optimizer.step(): 根据计算出的梯度更新模型的参数(这里使用的是**SGD优化器**)。
> * output.item(): 返回值的事损失值，表示训练过程中的误差大小。
### 2.4 主函数
> ```Python
> def main():
>     torch.manual_seed(42)
>     X = torch.linspace(-1, 1, 101)
>     Y = 2 * X + torch.randn(X.size()) * 0.33
>     model = build_model()
>     loss = torch.nn.MSELoss(reduction='elementwise_mean')
>     optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9)
>     batch_size = 10
>     for i in range(100):
>         cost = 0.
>         num_batchs = len(X) // batch_size
>         for k in range(num_batches):
>             start, end = k * batch_size, (k+1) * batch_size
>             cost += train(model, loss, optimizer, X[start:end], Y[start:end])
>         print("Epoch= %d, cost = %s",i+1, cost/ num_batches)
>     w = next(model.parameters()).data
>     print("w = %.2f" % w.numpy())
> ```
> * torch.manual_seed(42): 设置随机种子，确保每次运行时结果相同，便于调试
> * X = torch.linspace(-1, 1, 101): 生成一个从-1到1之间均匀分布的101个点
> * Y = 2 * X + torch.randn(X.size()) * 0.33: 生成目标值Y，它是X的线性变换，添加了高斯噪声(标准差为0.33)。 相当于想学习的线性回归模型的目标输出。
> * model = build_model(): 构建线性回归模型。
> * loss = torch.nn.MSELoss(reduction='elementwise_mean'): 选择损失函数，这里是**均方误差(MSE)**。reduction='elementwise_mean'表示计算所有样本的平均损失。
> * optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9): 选择优化器，这里使用**随机梯度下降(SGD)**，学习率为0.01，动量为0.9。
> * batch_size = 10: 定义每批次训练的样本数。
> * for i in range(100): 进行100次训练(即100个Epoch)。
> * num_batches = len(x) // batch_size: 计算每个Epoch需要的批次数量。
> * for k in range(num_batches): 按批次训练模型。每次使用X[start:end]和Y[start:end]进行训练，并累加损失。
> * print("Epoch= %d, cost = %s",i+1, cost/ num_batches): 打印每个Epoch的平均损失。
> * w = next(model.parameters()).data: 获取模型的第一个(也是唯一一个)参数w(模型的权重)。
> * print("w = %.2f" % w.numpy()): 打印最终学到的权重值。
