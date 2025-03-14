# PyTorch 简介
> PyTorch 是一个灵活且高效的深度学习框架，提供动态图计算、自动求导 和 GPU 加速，广泛应用于研究和生产，其书张量计算API与numpy类似。
# PyTorch 白手入门
> 1. 判断cuda是否可用
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
> 2. 简单线性回归
> 2.1 导入必要的库
> ```Python
> import torch
> from torch import optim
> ```
> * torch: Pytorch的核心库，提供对张量(Tensor)运算的支持。
> * optim: PyTorch中的优化器库，提供了各种优化算法(如梯度下降、Adam等)。
> 2.2 构建模型
> ```Python
> def build_model():
>     model = torch.nn.Sequential()
>     model.add_module("linear",torch.nn.Linear(1,1,bias=False))
> ```
> * torch.nn.Sequential(): 一个容器，按顺序添加多个层，构建神经网络模型。在这个例子中，我们只用一个线性层。
> * torch.nn.Linear(1,1,bias=False): 这是一个全连接层(线性层)，输入和输出的维度都是1。bias=False表示的是没有偏置项(通常在线性回归中，偏置项是可选的)。该层的的作用是计算线性变换 y=wx+b 偏置项b被禁用了
> 3. 训练函数
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
