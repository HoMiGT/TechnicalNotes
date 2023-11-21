# 查看磁盘设备名
```Shell
# lsblk
或者
# fdisk -l 
```

# 创建分区
```Shell
# fdisk /dev/vdb
m: help
n: 创建一个分区
p: 创建一个主分区
w: 写入分区信息
依次输入  n   p  w 
```

# 创建ext4文件系统
```Shell
# mkfs.ext4 /dev/vdb
```

# 创建目录并挂载同时授予admin用户权限
```Shell
# mkdir /mnt/mydisk
# mount /dev/vdb /mnt/mydisk
# chown -R admin:admin /mnt/mydisk
```
