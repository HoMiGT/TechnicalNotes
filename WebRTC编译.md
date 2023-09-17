WebRTC安装步骤
1. 安装 depot_tools
```shell

git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git

```

2. 配置环境变量
```shell
export PATH=$PATH:/home/admin/tools/depot_tools
```

3. 下载webrtc源码
```shell
mkdir webrtc
cd webrtc
fetch --nohooks webrtc
```

4. 编辑.gclient文件
```shell
vim .gclient

solutions = [
  { 
    "name"     : "src",  
    "url"      : "https://webrtc.googlesource.com/src", 
    "managed"  : False,  
    "custom_deps": {},  
  },
]

```

5. 运行gclient sync 同步源码
```shell
date;gclient sync;date
```

6. 编译源码
```shell
sudo dnf install epel-release redhat-lsb-core
./build/
```


备注：
  [开源库](https://ftp.gnu.org/gnu) 
