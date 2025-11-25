# windows代理linux的请求
* 启动ip服务
```PowerShell
net start iphlpsvc
```
* 监听流量转给本机地址
```PowerShell
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=17897 connectaddress=127.0.0.1 connectport=7897
```
* 限定局域网内IP可访问
```PowerShell
New-NetFirewallRule -DisplayName "Allow-Port-17897-LAN" -Direction Inbound -LocalPort 17897 -Protocol TCP -RemoteAddress "192.168.199.0/24" -Action Allow
```
* 查看开通情况
```PowerShell
netsh interface portproxy show all
```
