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
## 3. 简单的逻辑回归
### 3.1 导入必要的库
> ```Python
> import numpy as np
> import torch
> from torch import optim
> from data_util import load_mnist
> ```
> data_util.py
> ```
> import gzip
> import os
> from os import path
> import numpy as np
> from torchvision import datasets, transforms
>
> DATASET_DIR = 'datasets/MNIST/raw'
> MNIST_FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
>               "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
> def one_hot(x,n):
>     if type(x) == list:
>         x = np.array(list)
>     x = x.flatten()
>     o_h = np.zeros((len(x),n))
>     o_h[np.arange(len(x)),x] = 1
>     return o_h
>
> def download_mnist():
>     for f in MNIST_FILES:
>         f_path = os.path.join(DATASET_DIR,f)
>         if not path.exists(f_path):
>             transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,),(1,))])
>             train_set = datasets.MNIST(root="./datasets", train=True, download=True, transform=tansform)
>             test_set = datasets.MNIST(root="./datasets", train=False, download=True, transform=transform)
> 
> def load_mnist(n_train=60000, n_test=10000, onehot=True):
>     checks = [path.exists(os.path.join(DATASET_DIR,f)) for f in MNIST_FILES]
>     if not np.all(checks):
>         download_mnist()
>     with gzip.open(os.path.join(DATASET_DIR, "train-images-idx3-ubyte.gz")) as fd:
>         buf = fd.read()
>         loaded = np.frombuffer(buf, dtype=np.uint8)
>         tr_X = loaded[16:].reshape((60000,28*28)).astype(float)
>
>     with gzip.open(os.path.join(DATASET_DIR, "train-labels-idx1-ubyte.gz")) as fd:
>         buf = fd.read()
>         loaded = np.frombuffer(buf,dtype=np.uint8)
>         tr_Y = loaded[8:].reshape(60000)
>
>     with gzip.open(os.path.join(DATASET_DIR, "t10k-images-idx3-ubyte.gz")) as fd:
>         buf = fd.read()
>         loaded = np.frombuffer(buf,dtype=np.uint8)
>         te_X = loaded[16:].reshape((10000, 28*28)).astype(float)
>
>     with gzip.open(os.path.join(DATASET_DIR, "t10k-labels-idx1-ubyte.gz")) as fd:
>         buf = fd.read()
>         loaded = np.frombuffer(buf,dtype=np.uint8)
>         te_Y = loaded[8:].reshape(10000)
>
>     tr_X = tr_X[:n_train]
>     tr_Y = tr_Y[:n_train]
>     te_X = te_X[:n_test]
>     te_Y = te_Y[:n_test]
>
>     if onehot:
>         tr_Y = one_hot(tr_Y, 10)
>         te_Y = one_hot(te_Y, 10)
>     else:
>         tr_Y = np.asarray(tr_Y)
>         te_Y = np.asarray(te_Y)
>     return tr_X, te_X, tr_Y, te_Y
> ```
### 3.2 构建模型
> ```Python
> def build_model(input_dim, output_dim):  # input_dim是28*28的784维向量，output_dim有10个类别 0-9
>     model = torch.nn.Sequential()  
>     model.add_module("linear", torch.nn.Linear(input_dim, output_dim, bias=False))  # 逻辑回归本质是 线性回归 y = Wx 不适用偏置
>     return model
> ```
### 3.3 训练模型
> ```Python
> def train(model, loss, optimizer, x_val, y_val):
>     model.train()  # 设定为训练模式
>     x = x_val.requires_grad_(False)  # 不计算梯度(加速计算)
>     y = y_val.requires_grad_(False)
>     optimizer.zero_grad()   # 清除上一次的梯度
>     fx = model.forward(x)   # 前向传播
>     output = loss.forward(fx, y)  # 计算损失
>     output.backward()  # 反向传播
>     optimizer.step()  # 更新参数
>     return output.item()
> ```
### 3.4 预测模型
> ```Python
> def predict(model, x_val):
>     model.eval()  # 设定为预测模式，评估模式
>     x = x_val.requires_grad_(False)  
>     output = model.forward(x)
>     return output.data.numpy().argmax(axis=1)   # 取每行最大值对应的索引(即预测类别)
> ```
### 3.5 主函数
> ```Python
> def main():
>     torch.manual_seed(42)
>     tr_X, te_X, tr_Y, te_Y = load_mnist(onehot=False)  # 加载模型
>     tr_X = torch.from_numpy(tr_X).float()
>     te_X = torch.from_numpy(te_X).float()
>     tr_Y = torch.from_numpy(tr_Y).long()
>
>     n_examples, n_features = tr_X.size()
>     n_classes = 10   # 手写数字 10个类别
>     model = build_model(n_features, n_examples)
>     loss = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')   # 用于分类任务 默认softmax+NLLLoss
>     optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9)   # 随机梯度下降
>     batch_size=100
>     for i in range(100):  # 训练100个Epoch 
>         cost = 0.
>         num_batches = n_examples // batch_size  # 计算批次数
>         for k in range(num_batches):
>             start, end = k*batch_size, (k+1)*batch_size
>             cost += train(model, loss, optimizer, tr_X[start:end], tr_Y[start:end])  # 训练
>         pred_Y = predict(model, te_X)  # 预测
>         print(f"Epoch {i+1}, cost = {cost/num_batches}, acc= {100. * np.mean(pred_Y == te_Y)}") 
> ```
### 3.6 汇总
> 1. 数据准备
>    * 用torch.from_numpy()转换数据
>    * float()处理图像，long()处理分类标签
> 2. 构建模型
>    * torch.nn.Linear()定义神经网络结构
>    * torch.nn.Sequential()组织多个层
> 3. 定义损失函数&优化器
>    * CrossEntropyLoss()计算损失
>    * SGD()进行参数优化
> 4. 训练循环
>    * zero_grad()清除梯度
>    * forward()计算前向传播
>    * backward()计算梯度
>    * step()更新参数
> 5. 模型评估
>    * model.eval()关闭dropout
>    * argmax()取预测类别
### 3.7 更深一步
> 深度学习：增加 **隐藏层**(ReLU()+Linear())
> ```Python
> model = torch.nn.Sequential(
>   torch.nn.Linear(784,128)  # 隐藏层1
>   torch.nn.ReLU(),
>   torch.nn.Linear(128,64)  # 隐藏层2
>   torch.nn.ReLU(),
>   torch.nn.Linear(64,10)  # 输出层
> )
> ```
## 4. 神经网络
### 4.1 导入必要的库
> ```
> import numpy as np
> import torch
> from torch import optim
> from data_util import load_mnist
> ```
### 4.2 构建模型
> ```
> def build_model(input_dim, output_dim):
>     model = torch.nn.Sequential()
>     model.add_module("linear_1",torch.nn.Linear(input_dim,512,bias=False))
>     model.add_module("sigmoid_1",torch.nn.Sigmoid())
>     model.add_module("linear_2",torch.nn.Linear(512,output_dim,bias=False))
>     return model
> ```
> 俩层神经网络,输入input_dim的特征数量(MNIST是28*28=784)，output_dim输出类别数(MNIST有10个数字0-9)
> 神经网络结构:
>     * 第一层 线性层，输入是784维(28*28),输出是512维
>     * 激活函数：Sigmoid(), 给上一层加一个非线性变换，让神经网络能学到更复杂的特征(通常更推荐ReLU)
>     * 第二层 输出层, 输入512维输入，输出10维(对应0-9的概率分布)
### 4.3 训练模型
> ```
> def train(model,loss,optimizer,x_val,y_val):
>     model.train()  # 让模型进入训练模式
>     x = x_val.requires_grad_(False)
>     y = y_val.requires_grad_(False)
>     optimizer.zero_grad()  # 每次迭代前清空梯度，防止梯度累加
>     fx = model.forward(x)   # 前向传播，fx 是模型的预测结果(一个行为[batch_size,10]的张量)
>     output = loss.forward(fx,y)  # 计算损失，CrossEntroyLoss会计算预测值和真实标签y之间的差距
>     output.backward()  # 反向传播梯度
>     optimizer.step()  # 更新参数,让模型的参数朝着降低损失的方向调整
>     return output.item()  
> ```
### 4.4 预测模型
> ```
> def predict(model,x_val):
>     model.eval()  # 设定模型为评估模型 影响BatchNorm、Dropout等
>     x = x_val.requires_grad_(False) 
>     output = model.forward(x)  # 计算模型的输出
>     return output.data.numpy().argmax(axis=1)  # 取最大值作为预测类别
> ```
### 4.5 主函数
> ```
> def main():
>     torch.manual_seed(42)  # 设置随机种子，保证结果可复现
>     tr_X, te_X, tr_Y, te_Y = load_mnist(onehot=False)  # 载入数据
>     tr_X = torch.from_numpy(tr_X).float()
>     te_X = torch.from_numpy(te_X).float()
>     tr_Y = torch.from_numpy(tr_Y).float()
>     n_examples, n_features = tr_X.size()  # 获取数据形状
>     n_classes = 10  # MNIST 任务有10个类别
>     model = build_model(n_features, n_classes)   # 创建模型
>     loss = torch.nn.CrossEntropyLoss(reduction="elementwise_mean")  # 定义损失函数  
>     optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9)   # 定义优化器
>     batch_size = 100   # 批次大小
>     for i in range(100):  # 训练100轮
>         cost = 0.
>         num_batches = n_examples // batch_size
>         for k in range(num_batches):
>             start, end = k * batch_size, (k+1) * batch_size
>             cost += train(model, loss, optimizer, tr_X[start:end], tr_Y[start:end])
>         pred_Y = predict(model, te_X)
>         print(f"Epoch {i+1}, cost= {cost/num_batches}, acc={100. * np.mean(pred_Y == te_Y)}")
> ```
## 5. 现在深度学习
### 5.1 导入必要的库
> ```
> import numpy as np
> import torch
> from torch import optim
> from data_util import load_mnist
> ```
### 5.2 构建模型
> ```
> def build_model(input_dim,output_dim):
>     model = torch.nn.Sequential()
>     model.add_module("linear_1",torch.nn.Linear(input_dim,512,bias=False))
>     model.add_module("relu_1",torch.nn.ReLU())
>     model.add_module("dropout_1",torch.nn.Dropout(0.2))
>     model.add_module("linear_2",torch.nn.Linear(512,512,bias=False))
>     model.add_module("relu_2",torch.nn.ReLU())
>     model.add_module("dropout_2",torch.nn.Dropout(0.2))
>     model.add_module(linear_3,torch.nn.Linear(512,output_dim,bias=False))
>     return model
> ```
> 模型是一个三层神经网络
>   * 第一层： 线性变换 Linear(748,512), 输入是784(28x28)
>   * ReLU激活：ReLU()非线性函数，增加模型的表达能力。解决Sigmoid **梯度消失** 问题，训练更稳定、收敛更快
>   * Dropout(0.2): 防止过拟合，每次训练随机丢弃20%的神经元
>   * 第二层：Linear(512,512) -> ReLU() -> Dropout(0.2) 继续进行特征提取
>   * 第三层: Linear(512,10), 输出10维 表示0~9 
### 5.3 训练模型
> ```
> def train(model,loss,optimizer,x_val,y_val):
>     model.train()  # 设置为训练模式  让Dropout() 生效，确保训练时随机丢弃神经元
>     x = x_val.requires_grad_(False)
>     y = y_val.requires_grad_(False)
>     optimizer.zero_grad()   # 清空梯度
>     fx = model.forward(x)   # 前向传播
>     output = loss.forward(fx,y)  # 计算损失
>     output.backward()  # 反向传播计算梯度
>     optimizer.step()  # 更新参数  使用Adam优化器调整权重，让模型更接近真实标签
>     return output.item()
> ```
### 5.4 预测模型
> ```
> def predict(model,x_val):
>     model.eval()  # 让Dropout() 失效(即不再随机丢弃神经元)
>     x = x_val.requires_grad_(False)  
>     output = model.forward(x)  # 计算测试集的预测结果
>     return output.data.numpy().argmax(axis=1)
> ```
### 5.5 主函数
> ```
> def main():
>     torch.manual_seed(42)   # 固定随机种子，保证结果可复现
>     tr_X, te_X, tr_Y, te_Y = load_mnist(onehot=False)  # 载入数据
>     tr_X = torch.from_numpy(tr_X).float()
>     te_X = torch.from_numpy(te_X).float()
>     tr_Y = torch.from_numpy(tr_Y).float()
>     n_examples, n_features = tr_X.size()  # 获取数据形状
>     n_classes = 10  # MNIST 有10类
>     model = build_model(n_features,n_classes)  # 创建神经网络
>     loss = torch.nn.CrossEntropyLoss(reduction="elementwise_mean")  # 交叉熵损失函数
>     optimizer = optim.Adam(model.parameters())  # Adam优化器
>     batch_size = 100  # 每批次大小
>     for i in range(100):  # 训练100轮
>         cost = 0.
>         num_batches = n_examples // batch_size
>         for k in range(num_batches):
>             start, end = k * batch_size, (k+1)*batch_size
>             cost += train(model,loss,optimizer,tr_X[start:end],tr_Y[start:end])
>             pred_Y = predict(model, te_X)
>         print(f"Epoch: {i+1}, cost = {cost / num_batches}, acc = {100. * np.mean(pred_Y == te_Y)}")
> ```



