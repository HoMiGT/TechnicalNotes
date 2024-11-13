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
> 解压压缩包: tar -xvzf TensorRT-
> 
# 3. TensorRT的CMake工程
> * 对于cuda，可以使用find_package函数来查找
> ```
> find_package(CUDA REQUIRED)
> message(STATUS "Found CUDA headers: ${CUDA_INCLUDE_DIRS}")
> message(STATUS "Found CUDA libraries: ${CUDA_LIBBARIES}")
> ```
> * 对于TensorRT，暂不支持find_package,需要借助find_path来查找头文件
> ```
> set(TENSORRT_ROOT /usr/local/TensorRT-8.2.5.1)
> find_path(TENSORRT_INCLUDE_DIR NvInfer.h
>   HINTS ${TENSORRT_ROOT} PATH_SUFFIXES include/
> )
> message(STATUS "Found TensorRT headers: ${TENSORRT_INCLUDE_DIR}")
> ```
> * 以及要用find_package来查找库文件
> ```
> find_library(TENSORRT_LIBRARY_INFER nvinfer
>    HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
>    PATH_SUFFIXES lib lib64 lib/x64
> )
> message(STATUS "Found TensorRT-nvinfer libs: ${TENSORRT_LIBRARY_INFER}")
> ```
> 其中TENSORRT_ROOT为TensorRT的实际安装目录。之后可以使用target_include_directories与target_link_libraries引用和链接正确的头文件与库文件。
> 
