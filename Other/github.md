# 修改主机hosts
```shell
# 分别查询  
github.com
github.global.ssl.fastly.net
assets-cdn.github.com
dl-ssl.google.com
groups.google.com
ajax.googleapis.com
# 的dns对应的ip地址



# 修改 /etc/hosts
vim /etc/hosts

20.205.243.166 github.com
39.109.122.128 github.global.ssl.fastly.net
185.199.111.153 assets-cdn.github.com
64.233.189.136 dl-ssl.google.com
103.39.76.66 groups.google.com
142.251.43.10 ajax.googleapis.com


# 重启network
/etc/init.d/network restart
```

# 国内加速下载github库
```shell
git config --global url."https://github.com.cnpmjs.org/".insteadOf "https://github.com/"

# 这条命令将全局的 Git 配置更改为使用 https://github.com.cnpmjs.org/ 作为 GitHub 的镜像地址。
# 需要注意的是，这个方法只能提高下载速度，如果你需要推送代码到 GitHub，还是需要正常的 GitHub 地址。

# fastgit.org
# https://doc.fastgit.org/

# gitclone.com
# https://gitclone.com/

# cnpmjs.org
# https://github.com.cnpmjs.org/

```




