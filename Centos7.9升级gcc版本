Centos7.9默认版本是4.8.5，升级步骤如下

#### 1.安装centos-release-scl
  # yum install centos-release-scl

#### 2.安装devtoolset，注意，如果想安装7.*版本的，就修改成devtoolset-7-gcc*
  # yum install devtoolset-8-gcc*
  
#### 3.激活对应的devtoolset
  # scl enable devtoolset-8 bash
  
#### 4.查看对应升级版本
  # gcc -v
  
说明：安装的devtoolset是在/opt/rh目录下，每个版本的目录下面都有个enable文件，如需启动某个版本，只需执行 source ./enable

#### 5.直接替换旧版的gcc
  # mv /usr/bin/gcc /usr/bin/gcc-4.8.5

  # ln -s /opt/rh/devtoolset-8/root/bin/gcc /usr/bin/gcc

  # mv /usr/bin/g++ /usr/bin/g++-4.8.5

  # ln -s /opt/rh/devtoolset-8/root/bin/g++ /usr/bin/g++
  
  # mv /usr/bin/c++ /usr/bin/c++-4.8.5
  
  # ln -s /opt/rh/devtoolset-8/root/bin/c++ /usr/bin/c++
  
  # mv /usr/bin/cc /usr/bin/cc-4.8.5
  
  # ln -s /opt/rh/devtoolset-8/root/bin/cc /usr/bin/cc

  # gcc --version

  # g++ --version
