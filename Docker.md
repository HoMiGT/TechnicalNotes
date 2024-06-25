# Docker 迁移
## 镜像
> * 归档文件
> ```Shell
> sudo docker save [image-id | image-name] | gzip > <name>.tar.gz
> sudo docker save [image-id | image-name]  >  <name>.tar
> sudo docker save -o <name>.tar [image-id | image-name]
> ```
> * 加载文件
> ```Shell
> sudo docker load -i <name>.tar
> sudo docker load < <name>.tar
> ```
## 容器
> * 导出
> ```Shell
> sudo docker export [container-id | container-name] | gzip > <name>.tar.gz
> sudo docker export [container-id | container-name] > <name>.tar
> sudo docker export -o <name>.tar [container-id | container-name] 
> ```
> * 导入
> ```Shell
> sudo docker import <name>.tar <new-name>:<tag>
> ```
# Docker container2image
> * 提交
> ```Shell
> sudo docker commit [container-id | container-name] <image-name>:<tag>
> ```
# Docker 管理
> * 直接使用docker命令管理
> ```Shell
> docker run --rm -it --gpu all -u admin -v /home/admin/projects/alsm:/home/admin/projects/alsm -p 8000-9000:8000-9000 [image-id | image-name] /bin/bash
> ```
> * 使用docker-compose管理 [docker-compose release](https://github.com/docker/compose/releases)
> ```Shell
> # curl -L "https://github.com/docker/compose/releases/download/v2.27.3/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
> # chmod +x /usr/local/bin/docker-compose
> # ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose
> # docker-compose --version
> 删除docker-compose 
> # rm -r /usr/local/bin/docker-compose
> 
> docker-compose 命令
> 查看帮助
> $ docker-compose -h
> 启动所有docker-compose服务
> $ docker-compose up
> 启动所有docker-compose服务 后台运行
> $ docker-compose up -d
> 停止并删除容器、网络、卷、镜像  
> $ docker-compose down
> ....... 待完善
> ```
> * docker 清理磁盘
> ```Shell
> 1. 清理docker系统中不再使用的镜像、缓存、容量和网络等资源，删除未使用的镜像，停止的容器，无效的网络，释放磁盘空间
> docker system prune -a
> ```
> * docker 迁移目录
> ```Shell
> # docker info | grep "Docker Root Dir"
> # systemctl stop docker
> # systemctl status docker
> # rsync -avzP /var/lib/docker /home/docker_workspace/
> # mv /var/lib/docker /var/lib/docker.bak
> # ln -s /home/docker_workspace/docker /var/lib/docker
> # systemctl start docker
> # systemctl status docker
> # rm -rf /var/lib/docker.bak
> ```
