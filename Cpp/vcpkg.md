# 设置windows使用clang编译
* 下载安装 [msys2](https://www.msys2.org/)

* 安装mingw
```
pacman -S mingw-w64-x86_64-gcc
```
* 在vcpkg/triplets下创建文件
```x64-mingw-clang
set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)

set(VCPKG_CMAKE_C_COMPILER "E:/MSYS2/mingw64/bin/clang.exe")
set(VCPKG_CMAKE_CXX_COMPILER "E:/MSYS2/mingw64/bin/clang++.exe")

set(CMAKE_LINKER "E:/MSYS2/mingw64/bin/ld.exe")

set(VCPKG_CMAKE_FIND_ROOT_PATH "E:/MSYS2/mingw64")
set(VCPKG_CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(VCPKG_CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(VCPKG_CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# 使用MinGW的标准库
set(CMAKE_C_FLAGS "-fuse-ld=mingw -static-libgcc --static-libstdc++")
set(CMAKE_CXX_FLAGS "-fuse-ld=mingw -static-libgcc -static-libstdc++")
```
* 修改环境变量
```
CC :  D:\Softwares\llvm\bin\clang.exe
CXX : D:\Softwares\llvm\bin\clang++.exe
VCPKG_CC :  clang
VCPKG_CXX : clang++
VCPKG_ROOT : E:\vcpkg
VCPKG_OVERLAY_TRIPLETS : E:\vcpkg\triplets
VCPKG_DEFAULT_TRIPLET : x64-mingw-clang

```
