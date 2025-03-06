# 1. 最简单的项目
> ```CMakeLists.txt
> cmake_minimum_required(VERSION 3.25)  # 指定cmake的版本
> project(projectName VERSION 1.0)  # 指定项目名称 项目版本 
> add_executable(projectName main.cpp)
> ```
# 2. 设置c++的标准
> ```CMakeLists.txt
> set(CMAKE_CXX_STANDARD 20)  # 设置编译期c++的版本
> set(CMAKE_CXX_STANDARD_REQUIRED ON) # 强制使用编译器c++标准
> set(CMAKE_CXX_EXTENSIONS OFF)  # 关闭编译器c++特殊扩展 使用 -std=c++20 标志（而不是 -std=gnu++20），禁用 GNU 扩展。
> ```
# 3. 目前clang19.1.1仅支持自定义的module
> ```CMakeLists.txt
> set(CMAKE_CXX_SCAN_FOR_MODULES ON)  # 启用C++20模块的依赖扫描功能
> set(MODULES_SOURCES moduleDirectory/module.ixx)  # 设置ixx路径
> # 专为c++20module设置 添加关联模块文件
> target_sources(projectName PUBLIC
>     FILE_SET CXX_MODULES
>     BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
>     FILES ${MODULE_SOURCES}  
> )
> ```
