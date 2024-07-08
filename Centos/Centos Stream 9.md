# 安装Cuda 
> ## 安装步骤
>> [cuda guide install](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
>> [download latest cuda](https://developer.nvidia.com/cuda-downloads)     
>> ``` Shell
>> 1. 查看是否有nvidia驱动
>> # lspci | grep -i nvidia
>> 2. 发行版和版本号
>> # uname -m && cat /etc/*release
>> 3. 查看系统内核
>> # uname -r
>> 4. 更新系统并安装内核头文件
>> # dnf update
>> # dnf install kernel-devel-$(uname -r) kernel-headers-$(uname -r)
>> 5. nvidia的驱动与系统自带的nouveau驱动冲突
>> # vim /etc/modprobe.d/blacklist-nouveau
>> `blacklist nouveau
>>  options nouveau modeset=0
>> `
>> 6. 查看禁用是否生效
>> # reboot
>> # lsmod | grep nouveau
>> 7. 下载 如 cuda_12.4.0_550.54.14_linux.run
>> # chmod +x cuda_12.4.0_550.54.14_linux.run && sh cuda_12.4.0_550.54.14_linux.run
>> 8. 查看是否安装成功
>> # nvidia-smi
>> 9. 安装桌面，无需求可忽略
>> # dnf groupinstall "Server with GUI"
>> # dnf groupinstall GNOME
>> 10. 启动图形界面
>> # startx
>> 11. 查看系统运行模式
>> # systemctl get-default
>> 12. 设置系统启动后进入文本界面
>> # systemctl set-default multi-user.target
>> 13. 设置系统启动后进入图形界面
>> # systemctl set-default graphical.target
>> # reboot 
>> ```
> ## 卸载步骤
>> ```Shell
>> 1. cuda 的卸载  X.Y 为cuda版本
>> # /usr/local/cuda-X.Y/bin/cuda-uninstaller
>> 2. nvidia 的卸载
>> # /usr/bin/nvidia-uninstall
>> ```
******
# 磁盘分区  没有尝试成功
> * 扩充root分区大小
> ```Shell
> 1. 查看磁盘空间大小
> # df -h
> 2. 备份home目录文件
> # cp -r /home/ homebak/
> 3. 卸载home分区
> # umount /home && df -h
> 4. 删除home所在lv
> # lvremove /dev/mapper/cs-home
> 5. 扩展root所在lv
> # lvextend -L +200G /dev/mapper/cs-root
> # df -h
> 6. 拓展文件系统，并查看分区是否扩展成功
> # xfs_growfs /dev/mapper/cs-root && df -h
> 7. 重新创建home的lv
> # lvcreate -L 667G -n home cs
> # df -h
> # vgdisplay
> 8. 创建home文件系统
> # mkfs.xfs /dev/mapper/cs-home
> 9. 挂载home分区
> # mount /dev/mapper/cs-home /home
> # df -h 
> ```
# ssh 
> * 配置ssh的超时时间
> ```Shell
> # vim /etc/ssh/sshd_config
> `
> TCPKeepAlive yes
> ClientAliveInterval 120
> ClientAliveCountMax 30
> `
> ```
