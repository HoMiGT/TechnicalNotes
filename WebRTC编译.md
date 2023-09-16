WebRTC安装步骤
1. 安装 depot_tools
```shell

git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git\

```

2. 配置环境变量
```shell
export PATH=$PATH:/home/admin/tools/depot_tools
```

3. 下载webrtc源码
```
mkdir webrtc
cd webrtc
fetch --nohooks webrtc
```
