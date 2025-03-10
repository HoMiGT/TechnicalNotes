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

# 8. 例子
## 8.1 简单自定义库以及引用库
> ```CMakeLists.txt
> cmake_minimum_required(VERSION 3.25)
> project(Calculator LANGUAGES CXX VERSION 1.0)
>
> add_library(calclib STATIC src/calclib.cpp include/calc/lib.hpp)
> target_include_directories(calclib PUBLIC include)
> target_compile_features(calclib PUBLIC cxx_std_11)
>
> add_executable(calc apps/calc/calc.cpp)
> target_link_libraries(calc PUBLIC calclib)
> ```
## 8.2 变量与缓存
> * 本地变量
> ```
> set(MY_VARIABLE "value")
> set(MY_LIST "one" "two")
> set(MY_LIST "one;two")
> ```
> 注意： 如果一个值没有空格，那么加和不加引号的效果一样。
> 当一个变量用${}引用，空格的解析规则和上述相同。而对于路径来说，路径很有可能会包含空格，因此 "${MY_PATH}" 更合适
> * 缓存变量
> ```
> # 如果一个变量未被定义，可如此声明
> set(MY_CACHE_VARIABLE "VALUE" CACHE STRING "Description") 
>
> # 可把变量作为一个临时的全局变量 
> set(MY_CACHE_VARIABLE "VALUE" CACHE STRING "" FORCE)  # 强制修改该变量的值
> mark_as_advanced(MY_CACHE_VARIABLE)  # cmake -L ..
>
> # 可实现同样效果的临时全局变量
> set(MY_CACHE_VARIABLE "VALUE" CACHE INTERNAL "")
> option(MY_OPTION "This is settable from the command line" OFF)
> 
> set_property(TARGET TargetName PROPERTY CXX_STANDARD 11)  # 一次设置一个属性
> set_target_properties(TargetName PROPERTIES CXX_STANDARD 11)  # 一次设置多个属性
>
> # 另一方式更加通用，可以一次性设置多个目标、
> get_property(ResultVariable TARGET TargetName PROPERTY CXX_STANDARD)
> ```
## 8.3 用CMake进行编程
> * 控制流程
> ```
> if (variable)
>   # If variable is `ON` `YES` `TRUE` `Y` or non zero number
> else()
>   # If variable is `0` `OFF` `NO` `FALSE` `N` `IGNORE` `NOTFOUND` `""`, or ends in `-NOTFOUND`
> endif()
> ```
> * 宏定义与函数
> ```
> function(SIMPLE REQUIRED_ARG)
>   message(STATUS "Simple arguments: ${REQUIRED_ARG}, followed by ${ARGN}")
>   set(${REQUIRED_ARG} "From SIMPLE" PARENT_SCOPE)
> endfunction()
> simple(This Foo Bar)
> message("Output: ${This}")
> 
> function(COMPLEX)
>   cmake_parse_arguments(
>     COMPLEX_PREFIX
>     "SINGLE;ANOTHER"
>     "ONE_VALUE;ALSO_ONE_VALUE"
>     "MULTI_VALUES"
>     ${ARGN}
>   )
> endfunction()
> ```
## 8.4 与代码进行交互
> * 通过CMake配置文件
> ```
> # Version.h.in
> #define MY_VERSION_MAJOR @PROJECT_VERSION_MAJOR@
> #define MY_VERSION_MINOR @PROJECT_VERSION_MINOR@
> #define MY_VERSION_PATCH @PROJECT_VERSION_PATCH@
> #define MY_VERSION_TWEAK @PROJECT_VERSION_TWEAK@
> #define MY_VERSION "@PROJECT_VERSION@"
> 
> configure_file(
>   "${PROJECT_SOURCE_DIR}/include/My/Version.h.in"
>   "${PROJECT_BINARY_DIR}/include/My/Version.h"
> )
> ```
> * 读入文件
> ```
> set(VERSION_REGEX "#define MY_VERSION[ \t]+\"(.+)\"")
> # 选择文件中与正则表达式相匹配的行
> file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/include/My/Version.hpp" VERSION_STRING REGEX ${VERSION_REGEX})
> string(REGEX REPLACE ${VERSION_REGEX} "\\1" VERSION_STRING "${VERSION_STRING}")
> project(My LANGUAGES CXX VERSION ${VERSION_STRING})
> ```
## 8.5 如何组织你的项目
> 首先，创建一个名为project项目，它有一个名为lib的库，有一个名为app的执行文件
> ```
> - project
>   - .gitignore
>   - README.md
>   - LICENCE.md
>   - CMakeLists.txt
>   - cmake
>     - FindSomeLib.cmake
>     - somethind_else.cmake
>   - include
>     - project
>       - lib.hpp
>   - src
>     - CMakeLists.txt
>     - app.cpp
>   - tests
>     - CMakeLists.txt
>     - testlib.cpp
>   - docs
>     - CMakeLists.txt
>   - extern
>     - googletest
>   - scripts
>     - helper.py
> ```
## 8.6 在CMake中运行其他程序
> * 在配置时运行一条命令
> ```
> find_package(Git QUIET)
>
> if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
>   execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
>                   WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
>                   RESULT_VARIABLE GIT_SUBMOD_RESULT)
>   if(NOT GIT_SUBMOD_RESULT EQUAL "0")
>     message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT},please checkout submodules")
>   endif()
> endif()
> ```
> * 在构建时运行一条命令
> ```
> find_package(PythonInterp REQUIRED)    # 找到python解释器
> # 添加自定义命令
> # 输出Generated.hpp 命令是通过python脚本，参数是依赖some_target
> add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/include/Generated.hpp"
>   COMMAND "${PYTHON EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/GenerateHeader.py" --argument DEPENDS some_target)
> # 添加自定义目标 显示构建该目标
> add_custom_target(generate_header ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/include/Generated.hpp")
> install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/Generated.hpp DESTINATION include)
> ```
> * CMake中常用的工具
> ```
> cmake -E <mode>  在CMakeLists.txt中常被写作 ${CMAKE_COMMAND} -E
> <mode> : copy(复制) make_directory(创建文件夹) remove(移除) create_symlink(基于Unix系统可用)
> ```
# 9 为CMake项目添加特性
> 默认的构建类型
> ```
> set(default_build_type "Release")
> if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
>   message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
>   set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
>   set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
> endif()
> ```
## 9.1 全局设置以及属性设置
> ```
> set(CMAKE_CXX_STANDARD 11 CACHE STRING "The C++ standard to use")  # 全局设置c++版本
> set(CMAKE_CXX_STANDARD_REQUIRED ON)  # 关闭cmake的回退功能，强制使用指定的c++版本
> set(CMAKE_CXX_EXTENSIONS OFF)  # 使用 -std=c++11 而不是 -std=gnu++11
>
> # 不想设置全局行为，也可以为每个目标设置属性以实现相同的效果
> set_target_properties(myTarget PROPERTIES
>                       CXX_STANDARD 11
>                       CXX_STANDARD_REQUIRED YES
>                       CXX_EXTENSIONS NO)
> ```
