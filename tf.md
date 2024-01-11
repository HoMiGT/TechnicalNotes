# 1. tf binary url 
> [tf_binary_so](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.13.0.tar.gz)
> 可直接通过c/c++或者rust来开发tensorflow

# 2. nvidia
## a. 显卡、cuda、cuda toolkit的关系
> 三者关系灵活，皆可灵活配置，需要注意的是 有些toolkit的开发是在选定版本开发的，因此，会有 >= cuda版本的要求     
> 三者关系：显卡提供硬件基础，cuda定义软件架构与平台，抽象层，cuda toolkit实现cuda平台到实际硬件的映射与工具支持，是连接和实现二者映射的桥梁。    
> **显卡** : 计算机中的一种重要组件，用于处理图形和图像相关的任务，是物理硬件。含*GPU*、*显存*、*显示接口*、*渲染管线*、*显卡驱动程序*、*显卡性能参数*等。    
> **cuda** : 是nvidia开发的并行计算平台和并行计算架构。可以看作是一个GPU计算的操作系统，定义了可在GPU硬件上运行的程序模型、指令集、内存模型等。安装gpu的驱动，实际就是安装对应的指令集，内存模型。    
> **cuda toolkit** : 实现cuda功能的软件开发工具包，提供了开发，编译，调试基于cuda的GPU应用的所有必要工具。主要包含：cuda驱动(访问硬件)，cuda运行时库(部署和运行cuda程序时需要链接库)，编译调试分析工具等。  
> 显卡，cuda，cuda toolkit 均可单独安装，需要注意 >= 版本。其中cuda toolkit里含有对应cuda驱动的安装，所以可以不用专门安装cuda的驱动。      
> [显卡，cuda，cuda toolkit版本对应](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)       
> [cuda toolkit下载](https://developer.nvidia.com/cuda-11-8-0-download-archive)   其中 cuda-11-8-0-download-archive，11-8-0可以替换成自己想要下载的对应版本。      
> 驱动可单独选择指定版本下载，**如果cuda toolkit安装了驱动就不需要此步骤了** [nvidia驱动](https://www.nvidia.cn/Download/index.aspx?lang=cn)   选择与硬件对应的版本。      
## b. 下载安装部署环境变量
> 下载安装cuda toolkit的步骤，默认会选择安装对应版本的驱动
> ```Shell
> wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
> chmod +x cuda_11.8.0_520.61.05_linux.run
> sudo bash cuda_11.8.0_520.61.05_linux.run
> ```    
> 亦可单独下载驱动
> ```Shell
> chmod +x NVIDIA-Linux-x86_64-545.29.06.run
> sudo bash NVIDIA-Linux-x86_64-545.29.06.run
> ```
