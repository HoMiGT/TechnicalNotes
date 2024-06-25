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
