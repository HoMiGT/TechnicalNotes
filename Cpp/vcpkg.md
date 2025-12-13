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

set(VCPKG_CMAKE_C_COMPILER "D:/Softwares/llvm/bin/clang.exe")
set(VCPKG_CMAKE_CXX_COMPILER "D:/Softwares/llvm/bin/clang++.exe")

set(VCPKG_CMAKE_FIND_ROOT_PATH "E:/MSYS2/mingw64")
set(VCPKG_CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(VCPKG_CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(VCPKG_CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)


# 可选：使用 lld 或特定 link flags
set(VCPKG_CMAKE_LINKER "D:/Softwares/llvm/bin/lld.exe")
set(VCPKG_C_FLAGS "-fuse-ld=lld" CACHE STRING "" FORCE)
set(VCPKG_CXX_FLAGS "-fuse-ld=lld" CACHE STRING "" FORCE)

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
