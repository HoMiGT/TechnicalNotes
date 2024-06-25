# 安装Cuda 
> ## 安装步骤
>> [cuda guide install](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)     
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
>> [download latest cuda](https://developer.nvidia.com/cuda-downloads)
>>
> ## 卸载步骤
>> ```Shell
>> 1. cuda 的卸载  X.Y 为cuda版本
>> # /usr/local/cuda-X.Y/bin/cuda-uninstaller
>> 2. nvidia 的卸载
>> # /usr/bin/nvidia-uninstall
>> ```
