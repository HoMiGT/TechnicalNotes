> 1. 启动镜像
>> 需要下载NVIDIA Docker依赖
>> ```
>> $ curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-contianer-toolkit.repo | sudo tee /etc/yum.repos.d/nvidia-contianer-toolkit.repo
>> $ sudo yum-config-mananger --enable nvidia-container-toolkit-experimental
>> $ sudo yum install -y nvidia-container-toolkit
>> $ sudo nvidia-ctk runtime configure --runtime=docker
>> $ sudo systemctl daemon-reload
>> $ sudo systemctl restart docker
>> ```
>> ```
>> docker run -it --gpus all -p 8010:8010 -p 8020:8020 -p 8030:8030 -p 8040:8040 -p 8042:8042 -p 8050:8050 -p 8060:8060 -p 8070:8070 -v /usr/local/TensorRT-8.6.1.6:/usr/local/TensorRT-8.6.1.6 --name alsm_gpu centos:7
>> ```
