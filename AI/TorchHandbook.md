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
> 
