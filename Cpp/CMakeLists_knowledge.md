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
> * 设置c++的编译版本
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
> * 地址无关代码 -fPIC
> ```
> set(CMAKE_POSITION_INDENPENDENT_CODE ON)  # SHARED 以及 MODULE 类型的库中会自动包含此标志，也可以显示声明 全局
> set_target_properties(lib1 PROPERTIES POSITION_INDENPENDENT_CODE ON)  # 对某个目标进行设置
> ```
> * Little libraries 在Linux -ldl 标志
> ```
> find_library(MATH_LIBRARY m)
> if(MATH_LIBRARY)
>   target_link_libraries(MyTarget PUBLIC ${MATH_LIBRARY})
> endif()
> ```
> * 程序间优化(Interprocedural optimization) -flto 
> ```
> include(CheckIPOSupported)
> check_ipo_supported(RESULT result)  # 检查是否支持
> if(result)
>   set_target_properties(foo PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)  # 支持则启动链接时间优化
> endif()
> ```
> * 一些实用工具
> ```
> find_program(CCACHE_PROGRAM ccache)
> if(CCACHE_PROGRAM)
>   set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")  # 封装目标编译
>   set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
> endif()
>
> find_program(CLANG_TIDY_EXE NAMES "clang-tidy" DOC "Path to clang-tidy executable")
> 
> ```
> * CMake中一些有用的模组
> ```
> # CMakeDependentOption
> include(CMakeDependentOption)
> cmake_dependent_option(BUILD_TESTS "Build your tests" ON "VAL1;VAL2" OFF)
> # 如上的代码是如下代码的缩写 
> if(VAL1 AND VAL2)
>   set(BUILD_TESTS_DEFAULT ON)
> else()
>   set(BUILD_TESTS_DEFAULT OFF)
> endif()
> option(BUILD_TESTS "Build your tests" ${BUILD_TESTS_DEFAULT})
> if(NOT BUILD_TESTS_DEFAULT)
>   mark_as_advanced(BUILD_TESTS)
> endif()
>
> # cmake_print_properties 轻松打印属性
> # cmake_print_variables  打印任意给定的变量的名称和值
>
> # CheckCXXCompilerFlag
> include(CheckCXXCompilerFlag)
> check_cxx_compiler_flag(-someflag OUTPUT_VARIABLE) # OUTPUT_VARIABLE 也会出现在打印的配置输出中
> # 还有 CheckIncludeFileCXX、CheckStructHasMember、TestBigEndian、CheckTypeSize
>
> # try_compile / try_run
> try_compile(RESULT_VAR bindir SOURCES source.cpp)  # 如果使用try_run，则将运行生成的程序的结果存储在 RUN_OUTPUT_VARIABLE 中
>
> # FeatureSummary
> include(FeatureSummary)  # 打印出找到的所有软件包以及你明确设定的所有选项
> set_package_properties(OpenMP PROPERTIES URL "http://www.openmp.org" DESCRIPTION "Parallel compiler directives" PURPOSE "This is what it does in my package") # 拓展包的默认信息
> add_feature_info(WITH_OPENMP OpenMP_CXX_FOUND "OpenMP (Thread safe FCNs only)")  # 添加任何选项使其成为 feature_summary 的一部分
> 
> if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
>   feature_summary(WHAT ENABLED_FEATURES DISABLED_FEATURES PACKAGES_FOUND)
>   feature_summary(FILENAME ${CMAKE_CURRENT_BINARY_DIR}/features.log WHAT ALL)
> endif()
> ```
> * CMake对IDE的支持
> ```
> # 用文件夹来组织目标(target)
> set_property(GLOBAL PROPERTY USE_FOLDERS ON)
> set_property(TARGET MyFile PROPERTY FOLDER "Scripts")  # 创建目标之后，为目标添加文件属性，将其目标MyFile归入到Scripts文件夹中，文件夹可以使用 / 进行嵌套
>
> # 用文件夹来组织文件
> source_group("Source Files\\New Directory" REGULAR_EXPRESSION ".*\\.c[ucp]p?")  # 传统方式
> source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/base/dir" PREFIX "Header Files" FILES ${FILE_LIST})  # ? 待理解
> ```
> * 调试代码
> ```
> message(STATUS "MY_VARIABLE=${MY_VARIABLE}")
> # 通过内建模组 更方便打印
> include(CMakePrintHelpers)
> cmake_print_variables(MY_VARIABLE)  # 打印一个变量
>
> cmake_print_properties(TARGETS my_target PROPERTIES POSITION_INDEPENDENT_CODE)  # 打印关于某些目标拥有的变量
>
> # 如果想知道构建项目cmake文件发生了什么
> cmake -S . -B build --trace-source=CMakeLists.txt --trace-expand   # 会打印出指定的文件运行在哪一行，且变量会直接展开它们的值
> ```
# 10. 包含子项目
## 10.1 子模组
> ```
> # 如果添加一个Git仓库，它与你的项目仓库使用相同的Git托管服务
> # git submodule add ../owner/repo.git extern/repo   # 使用相对于项目的相对路径
>
> # CMake的解决方案
> find_package(Git QUIET)
> if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
>   option(GIT_SUBMODULE "Check submodules during build" ON)
>   if(GIT_SUBMODULE)
>     message(STATUS "Submodule update")
>     execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} RESULT_VARIABLE GIT_SUBMOD_RESULT)
>     if(NOT GIT_SUBMOD_RESULT EQUAL "0")
>       message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
>     endif()
>   endif()
> endif()
> if(NOT EXISTS "${PROJECT_SOURCE_DIR}/extern/repo/CMakeLists.txt")
>   message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed.Please update submodules and try again.")
> endif()
>
> # 添加子项目
> add_subdirectory(extern/repo)
> ```
# 11. 导出与安装
> * 查找模块(不好的方式)
> ```Find<mypackage>.cmake 脚本是为了那些不支持CMake的库所设计。使用CMake的库，可以使用Config<mypackage>.cmake```
> * 添加子项目
> ```add_subdirectory 添加相应的子目录，适用于纯头文件和快速编译的库。```
> * 导出
> ```*Config.cmake```
## 11.1 安装
> 安装命令会将文件或目标"安装"到安装树中。
> ```
> install(TARGETS MyLib
>   EXPORT MyLibTargets
>   LIBRARY DESTINATION lib
>   ARCHIVE DESTINATION lib
>   RUNTIME DESTINATION bin
>   INCLUDES DESTINATION include
> )
> ```
> 给定CMake可访问的版本是个不错的方式。使用find_package时，可以这样指定版本信息
> ```
> include(CMakePackageConfigHelpers)
> write_basic_package_version_file(
>   MyLibConfigVersion.cmake
>   VERSION ${PACKAGE_VERSION}
>   COMPATIBILITY AnyNewerVersion
> )
> ```
> 接下来有俩个选择，创建MyLibConfig.cmake,可以直接将目标导入或手动写入，然后包含目标文件。若有依赖项如OpenMP,则需要添加相应的选项。
> 首先，创建一个安装目标文件(类似于在构建目录中创建文件)
> ```
> install(EXPORT MyLibTargets FILE MyLibTargets.cmake NAMESPACE MyLib:: DESTINATION lib/cmake/MyLib)
> ```
> 该文件将获取导出目标，并将其放入文件中。若没有依赖，只需要使用MyLibConfig.cmake代替MyLibTargets.cmake即可。
## 11.2 导出
> ```
> export(TARGETS MyLib1 MyLib2 NAMESPACE MyLib:: FILE MyLibTargets.cmake)  # CMakeLists.txt 文件的末尾，CMake可以在$HOME/.cmake/packages目录下找到导出的包
> 
> set(CMAKE_EXPORT_PACKAGE_REGISIRY ON) 
> export(PACKAGE MyLib)  # find_package(MyLib)就可以找到构建文件夹了。此方式有缺点：导入了依赖项，则需要再find_package之前导入它们。
> 
> ```
## 11.3 打包
> ```
> set(CPACK_PACKAGE_VENDOR "Vendor name")
> set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Some summary")
> set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
> set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
> set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
> set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENCE")
> set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
> ```
> ```
> include(CPack)
> ```
# 12. 查找库(或包)
## 12.1 CUDA
> 启动cuda语言
> ```
> project(MY_PROJECT LANGUAGES CUDA CXX)
>
> enable_language(CUDA)  # 支持可选的话，可以放在条件语句中
>
> include(CheckLanguage)
> check_language(CUDA)  # 检查cuda是可用
>
> CMAKE_CUDA_COMPILER # 通过检查该值，来看cuda开发包是否存在
> CMAKE_CUDA_COMPILER_ID  # 对于nvcc
> CMAKE_CUDA_COMPILER_VERSION  # 来检查cuda版本
> ```
> 设置cuda变量
> ```
> if(NOT DEFINED CMAKE_CUDA_STANDARD)
>   set(CMAKE_CUDA_STANDARD 11)
>   set(CMAKE_CUDA_STANDARD_REQUIRED ON)
> endif()
> ```
> 添加库/可执行文件
> ```
> set_target_properties(mylib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)  # 也可使用 CUDA_PTX_COMPILATION 属性创建一个 PTX（Parallel Thread eXecution）文件
> ```
> 内置变量
> ```
> CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES：指示cuda开发包内置Thrust等工具的目录
> CMAKE_CUDA_COMPILER: nvcc的具体路径
>
> set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
> cuda_select_nvcc_arch_flags(ARCH_FLAGS)  # 用户检查当前硬件的架构标志
> ```
## 12.2 OpenMP
> ```
> find_package(OpenMP)
> if(OpenMP_CXX_FOUND)
>   target_link_libraries(MyTarget PUBLIC OpenMP::OpenMP_CXX)
> endif()
> ```
## 12.3 Boost
> ```
> set(Boost_USE_STATIC_LIBS OFF)
> set(Boost_USE_MULTITHREADED ON)
> set(Boost_USE_STATIC_RUNTIME OFF)
> find_package(Boost 1.50 REQUIRED COMPONENTS filesystem)
> message(STATUS "Boost version: ${Boost_VERSION}")
> if(NOT TARGET Boost::filesystem)
>   add_library(Boost::filesystem IMPORTED INTERFACE)
>   set_property(TARGET Boost::filesystem PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${Boost_INCLUDE_DIR})
>   set_property(TARGET Boost::filesystem PROPERTY INTERFACE_LINK_LIBRARIES ${Boost_LIBRARIES})
> endif()
> target_link_libraries(MyExeOrLibrary PUBLIC Boost::filesystem)
> ```
## 12.4 MPI
> ```
> find_package(MPI REQUIRED)
> message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
> target_link_libraries(MyTarget PUBLIC MPI::MPI_CXX)
> ```
## 12.5 ROOT 高能物理学的C++工具包
> ```
> find_package(ROOT 6.16 CONFIG REQUIRED)
> add_executable(RootSimpleExample SimpleExample.cxx)
> target_link_libraries(RootSimpleExample PUBLIC ROOT::Physics)
> ```
