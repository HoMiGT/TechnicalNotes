# windows配置
以管理员权限打开PowerShell
```
Add-WindowsCapability -Online -Name OpenSSH.Server
Start-Service sshd
Set-Service -Name ssdh -StartupType 'Automatic'
```

# linux端配置
在终端启动
`# ssh -D 1080 your_user@windows_ip`
在.bashrc配置
```
export http_proxy=socks5://127.0.0.1:1080
```
即可通过curl下载数据
