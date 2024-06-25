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
> sudo docker export [contianer-id | contianer-name] | gzip > <name>.tar.gz
> sudo docker export [contianer-id | contianer-name] > <name>.tar
> sudo docker export -o <name>.tar [contianer-id | contianer-name] 
> ```
> * 导入
> ```Shell
> sudo docker import <name>.tar <new-name>:<tag>
> ```
