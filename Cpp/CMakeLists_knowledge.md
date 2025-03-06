来自于[cmake tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)       
[Modern CMake简体中文版](https://modern-cmake-cn.github.io/Modern-CMake-zh_CN/)
___
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
# 4. 添加库以及子文件夹
> 子文件夹的CMakeLists.txt
> ```CMakeLists.txt
> add_library(libName STATIC mysqrt.cxx)  # 添加库
> ```
> 根目录下的CMakeLists.txt
> ```CMakeLists.txt
> add_subdirectory(mysqrtDirectory)  # 添加子文件夹 ${PROJECT_SOURCE_DIR} 项目的源文件夹目录
> target_include_directories(projectName mysqrtDirectory/include)  # 将包含的目录添加到目标
> target_link_directories(projectName mysqrtDirectory/lib)  # 指定目标库的链接库目录
> ```
# 5. 添加库的使用要求
> ```CMakeLists.txt
> target_include_directories(libName INTERFACE ${CAMEK_CURRENT_SOURCE_DIR})   # 设置库包含当前源目录  ${CMAKE_CURRENT_SOURCE_DIR} 正在处理的源目录的路径
> list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/libName")   #  将libName目录下的文件拼接到EXTRA_INCLUDES变量中
>
> add_library(tutorial_compiler_flags INTERFACE)  # 添加一个接口库  多个项目都可以链接到 tutorial_compiler_flags 这个特征
> target_compile_features(tutorial_compiler_flags INTERFACE cxx_std_20)  # 给接口库添加编译器功能
> target_link_libraries(projectName PUBLIC libName tutorial_compiler_flags)
> ```
# 6. 添加生成器表达式
> ```CMakeLists.txt
> target_compile_options(tutorial_compiler_flags INTERFACE
>     "$<$<gcc_like_cxx>:-Wall;-Wextra;-Wshadow;-Wformat2;-Wunused>"
>     "$<$<msvc_cxx>:-W3>"
> )
> ```
# 7. 安装
> ```CMakeLists.txt
> install(FILES libName.h DESTINATION include)  # 安装头文件到include目录下
> install(TARGETS projectName DESTINATION bin)  # 安装可执行文件到bin目录下
>
> find_package(ThirdPartyLibrary REQUIRED)  # 找到依赖的第三方库
> get_target_property(THIRD_PARTY_LIB_PATH ThirdPartyLibrary IMPORTED_LOCATION)
> file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/lib)  # 创建库路径
> file(COPY ${THIRD_PARTY_LIB_PATH} DESTINATION ${CMAKE_BINARY_DIR}/lib)  # copy库到指定lib目录下
> ```
