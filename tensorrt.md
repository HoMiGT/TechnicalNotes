# 1. 安装cuda toolkit
> [cuda-toolkit downloads](https://developer.nvidia.com/cuda-downloads)
> 授权执行权限，并安装    
> 配置环境变量
> ```shell
> export PATH=/usr/local/cuda/bin:$PATH
> export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
> ```
> 刷新环境变量: source ~/.bashrc     
> 验证cuda安装：nvcc --version
# 2. 下载安装TensorRT
> [TensorRT SDK downloads](https://developer.nvidia.com/tensorrt)
> 解压压缩包: tar 
> 
