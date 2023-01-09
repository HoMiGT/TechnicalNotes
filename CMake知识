cmake - find_package

  官方库: cmake 官方预定义了依赖包的Module，以Mac通过brew安装的cmake为例，存储在 /usr/local/Cellar/cmake/3.25.1/share/cmake/Modules
        每个以 Find<PackageName>.cmake 命名的文件都对应着<PackageName>相应的包。可以通过CMakeLists.txt里的find_package()函数引用。
        
        假设项目中需要引入CURL库，添加如下代码在CMakeLists.txt即可

```          
find_package(CURL)
add_executable(curltest curltest.cc)
if(CURL_FOUND)
    target_include_directores(clib PRIVATE ${CURL_INCLUDE_DIR})
    target_link_libraries(curltest ${CURL_LIBRARY})
else(CURL_FOUND)
    message(FATAL_ERROR "CURL library not found")
endif(CURL_FOUND)
```

        对于系统预定义的 Find<PackageName>.cmake 模块，都会定义如下几个变量
          <PackageName>_FOUND :  表示模块是否被找到
          <PackageName>_INCLUDE_DIR / <PackageName>_INCLUDES : 库头文件
          <PackageName>_LIBRARY / <PackageName>_LIBRARIES : 动态库/静态库
  
  非官方库: find_package()仅对cmake编译的库有效
        不是通过cmake编译的库，需要先通过cmake库编译之后再使用
        
        例如 引入 glog库来进行日志记录，我们在 .../Module 的目录下未找到 FindGlog.cmake 因此需要我们自行安装glog库
        # 克隆该项目
        git clone https://github.com/google/glog.git
        # 切换到需要的版本
        cd glog
        git checkout v0.40
        # 根据官网的指南进行安装
        cmake -H. -Bbuild -G "Unix Makefiles" 
        cmake --build build
        cmake --build build --target install
        
        此时我们就可以通过与引入curl库一样的方式引入glog库了
```
find_package(glog)
add_executable(glogtest glogtest.cc)
if(GLOG_FOUND)
# 由于glog在链接时将头文件直接链接到了库里面
    target_link_libraries(glogtest glog::glog)
else(GLOG_FOUND)
    message(FATAL_ERROR "glog library not found")
endif(GLOG_FOUND)
```        
  原理: find_package 是如何找到我们的库的，有俩种模式，Module模式和Config模式。
        Module模式，也就是引入curl库的方式，FindCURL.cmake 将项目引入头文件路径和库文件路径。
                   搜索的路径有俩个，一个是 <path_cmake_installed>/share/cmake/Modules,另一个是我们指定的CMAKE_MODULE_PATH所在的目录
                   如果Module模式搜索失败，转入Config模式进行搜索
        Config模式，也就是引入glog库的方式，主要是通过 <PackageName>Config.cmake 或者 <lower-packagename>-config.cmake 找到并引入头文件路径和库文件路径
                   cmake文件所在路径在 /usr/local/lib/cmake 目录下
                   如安装的glog库，会在 /usr/local/lib/cmake/glog/ 目录下生成 glog-config.cmake 文件
                   
        综上所述，原生支持Cmake编译和安装的库通常会安装Config模式的配置文件到对应目录，这个配置文件直接配置来头文件、库文件的路径，以及各种cmake变量供find_package 使用。
        对于非由cmake编译的项目，我们通常会编写一个Find<PackageName>.cmake,通过脚本来获取头文件、库文件等信息。原生支持cmake的项目安装时会拷贝一份<PackageName>Config.cmake
        到系统目录中，因此在没有显示指定搜索路径时也可以顺利找到。
        
  如何编写Find<PackageName>.cmake模块
        首先我们在当前目录下新建一个ModuleMode的文件夹，里面我们编写一个计算俩个整数之和的一个简单的函数库。库函数以手工编写Makefile的方式进行安装，库文件安装在/usr/lib目录下，
        头文件放在/usr/include目录下。
        其中Makefile如下所示
          #1.准备工作，编译方式，目标文件名，依赖库路径的定义
          CC = g++
          CFLAGS := -Wall -O3 -std=c++11
          
          OBJS = libadd.o # .o文件与.cpp文件同名
          LIB = libadd.so # 目标文件名
          INCLUDE = ./ # 头文件目录
          HEADER = libadd.h # 头文件
          
          all : $(LIB)

          #2.生成.o文件
          $(OBJS) : libadd.cc
            $(CC) $(CFLAGS) -I ./ -fpic -c $< 0o $@
            
          #3.生成动态库文件
          $(LIB) : $(OBJS)
            rm -f $@
            g++ $(OBJS) -shared -o $@
            rm -f $(OBJS)
            
          #4.删除中间过程生成的文件
          clean:
            rm -f $(OBJS) $(TARGET) $(LIB)
            
          #5.安装文件
          install:
            cp $(LIB) /usr/lib
            cp $(HEADER) /usr/include
        
        编译安装
          make
          sudo make install 
        
        接下来我们回到我们的cmake项目中，在cmake的文件夹下新建一个FindAdd.cmake文件。我们的目标是找到库的头文件所在目录和共享文件所在的位置
        
          # 在指定目录下寻找头文件和动态文件的位置，可以指定多个目标路径
          find_path(ADD_INCLUDE_PATH libadd.h /usr/include/ /usr/local/include/ ${CMAKE_SOURCE_DIR}/ModuleMode)
          find_library(ADD_LIBRARY NAMES add PATHS /usr/lib/add /usr/local/lib/add ${CMAKE_SOURCE_DIR}/ModuleMode)
          if (ADD_INCLUDE_DIR AND ADD_LIBRARY)
            set(ADD_FOUND TRUE)
          endif(ADD_INCLUDE_DIR AND ADD_LIBRARY) 
        
        此时我们可以像引入curl一样引入我们自定义的库了
        在CMakeLists.txt中添加
          # 将项目目录下的cmake文件加入到CMAKE_MODULE_PATH中，让find_package能够找到我们自定义的库
          set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
          add_executable(addtest addtest.cc)
          find_package(ADD)
          if(ADD_FOUND)
            target_include_directories(addtest PRIVATE ${ADD_INCLUDE_DIR})
            target_link_libraries(addtest ${ADD_LIBRARY})
          else(ADD_FOUND)
            message(FATAL_ERROR "ADD library not found")
          endif(ADD_FOUND)
        

cmake - 自动导入头文件的函数

      #自动查找头文件路径函数(没有去重)
      macro(FIND_INCLUDE_DIR result curdir)  # 定义函数,2个参数:存放结果result；指定路径curdir；
          file(GLOB_RECURSE children "${curdir}/*.hpp" "${curdir}/*.h" )	# 遍历获取{curdir}中*.hpp和*.h文件列表
          message(STATUS "children= ${children}")	# 打印*.hpp和*.h的文件列表
          set(dirlist "")		# 定义dirlist中间变量，并初始化
          foreach(child ${children})	# for循环
              string(REGEX REPLACE "(.*)/.*" "\\1" LIB_NAME ${child})	# 字符串替换,用/前的字符替换/*h
              if(IS_DIRECTORY ${LIB_NAME})	# 判断是否为路径
                  LIST(APPEND dirlist ${LIB_NAME})	# 将合法的路径加入dirlist变量中
              endif()					# 结束判断
          endforeach()				# 结束for循环
          set(${result} ${dirlist})			# dirlist结果放入result变量中
      endmacro()					#  函数结束


      # 查找include目录下的所有*.hpp,*.h头文件,并路径列表保存到 INCLUDE_DIR_LIST 变量中
      FIND_INCLUDE_DIR(INCLUDE_DIR_LIST ${PROJECT_SOURCE_DIR})	 # 调用函数，指定参数
      
      
cmake - 引入外部项目

      直接将源码引入到项目中，编译自己的项目也会将第三方源码一同编译，特别使用git工具控制第三方代码的版本，防止本地安装库文件版本与项目冲突
      
      1. 通过Submodle的方式引入
      
        克隆spdlog为项目的子项目
        git submodule add https://github.com/gabime/spdlog.git
        
        本项目已经添加了submodule，要在项目根目录执行以下命令初始化
        git submodule init
        git submodule update
        
        切换到我们需要的版本
        git checkout v1.4.2
        
        我们已经clone好了，现在只需要将spdlog作为subdirectory加入CMakeLists.txt中即可
          
          project(ImportExternalProject)
          cmake_minimum_required(VERSION 3.5)
          
          add_definitions(-std=c++)
          add_subdirectory(spdlog)
          
      2. 在编译时下载项目并引入
          
         首先新建cmake目录，在目录下创建spdlog.cmake并加入以下内容
          
           include(ExternalProject)
           
           set(SPDLOG_ROOT ${CMAKE_BINARY_DIR}/thirdparty/SPDLOG)
           set(SPDLOG_GIT_TAG v1.4.1) # 指定版本
           set(SPDLOG_GIT_URL https://github.com/gabime/spdlog.git)
           # 指定配置指令 注意此处修改了安装目录，否则默认情况会安装到系统目录
           set(SPDLOG_CONFIGURE cd ${SPDLOG_ROOT}/src/SPDLOG $$ cmake -D CMAKE_INSTALL_PREFIX=${SPDLOG_ROOT} .) 
           # 指定编译指令 (需要覆盖默认指令，进入我们指定的SPDLOG_ROOT目录下)
           set(SPDLOG_MAKE cd ${SPDLOG_ROOT}/src/SPDLOG && make)
           # 指定安装指令 (需要覆盖默认指令，进入我们指定的SPDLOG_ROOT目录下)
           set(SPDLOG_INSTALL cd ${SPDLOG_ROOT}/src/SPDLOG && make install)
           
           ExternalProject_Add(SPDLOG
             PREFIX             ${SPDLOG_ROOT}
             GIT_REPOSITORY     ${SPDLOG_GIT_URL}
             GIT_TAG            ${SPDLOG_GIT_TAG}
             CONFIGURE_COMMAND  ${SPDLOG_CONFIGURE}
             BUILD_COMMAND      ${SPDLOG_MAKE}
             INSTALL_COMMAND    ${SPDLOG_INSTALL}
           )
           
           # 指定编译好的静态库文件的路径
           set(SPDLOG_LIBRARY ${SPDLOG}/lib/spdlog/libspdlog.a)
           # 指定头文件所在的目录
           set(SPDLOG_INCLUDE_DIR ${SPDLOG_ROOT}/include)
           
         以上我们指定来配置，编译，安装的指令。如果不这样做，cmake会默认将编译好的库和头文件安装在系统目录。而我们希望它安装在build
         目录下的指定位置。
         
         接下来，我们就可以在CMakeLists.txt中可以通过target_link_libraries 和target_include_directories进行引用
          
           project(ImportExternalProject)
           cmake_minimum_required(VERSION 3.5)
           
           add_definitions(-std=c++)
           add_executable(test_spdlog testspdlog.cc)
           
           set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
           include(spdlog)
           
           target_link_libraries(test_spdlog ${SPDLOG_LIBRARY})
           target_include_directories(test_spdlog PRIVATE ${SPDLOG_INCLUDE_DIR})
           
       3. 使用FetchContent(CMake 3.11+)
          
          步骤总结: 
            使用FetchContent_Declare(MyName)获取项目，可以是一个URL也可以是一个Git仓库
            使用FetchContent_GetProperties(MyName)获取我们需要的变量MyName_*
            使用add_subdirectory(${MyName_SOURCE_DIR} ${MyName_BINARY_DIR})引入项目。
            
          cmake 3.14版本，官方又为我们提供了更方便的FetchContent_MakeAvailable方法,将步骤2，3集成了一起。
     
        
          与前面类似，我们在cmake目录下新建spdlog2.cmake,使用FetchContent引入spdlog
            
            #添加第三方依赖包
            include(FetchContent)
            # FetchContent_MakeAvailable was not added until CMake 3.14
            if (${CMAKE_VERSION} VERSION_LESS 3.14)
              include(add_FetchContent_MakeAvailable.cmake)
            endif()
            
            set(SPDLOG_GIT_TAG v1.4.1)
            set(SPDLOG_GIT_URL https://github.com/gabime/spdlog.git)
            
            FetchContent_Declare(
              spdlog
              GIT_REPOSITORY  ${SPDLOG_GIT_URL}
              GIT_TAG         ${SPDLOG_GIT_TAG}
            )
            
            FetchContent_MakeAvailable(spdlog)
            
          在CMakeLists.txt中，包含cmake/spdlog2.cmake,便可将spdlog作为library来使用了
          
            project(ImportExternalProject)
            cmake_minimum_required(VERSION 3.11)
            
            add_definitions(-std=c++11)
            add_executable(test_spdlog testspdlog.cc)
            
            set(CMAK_MODULE_PATH "${CMAKE_SOURCE_DIR/cmake;${CMAKE_MODULE_PATH}")
            include(spdlog2)
            
            target_link_libraries(test_spdlog PRIVATE spdlog)
            
        
cmake - 如何编写自己的共享库

    将包含 mymath.h mymath.cpp mymathapp.cpp 的项目编写成对外共享的库
    
    在CMakeLists.txt添加如下配置
      
      cmake_minimum_required(VERSION 3.11)
      project(Installation VERSION 1.0)
      
      # 如果想生成静态库，使用下面的语句
      # add_library(mymath mymath.cpp)
      # target_include_directories(mymath PUBLIC ${CMAKE_SOURCE_DIR}/include)
      
      # 如果想生成动态库，使用下面的语句
      add_library(mymath SHARED mymath.cpp)
      target_include_directories(mymath PRIVATE ${CMAKE_SOURCE_DIR}/include)
      set_target_properties(mymath PROPERTIES PUBLIC_HEADER ${CMAKE_SOURCE_DIR}/include/mymath.h)
      
      # 生成可执行文件
      add_executable(mymathapp mymathapp.cpp)
      target_link_libraries(mymathapp mymath)
      target_include_directories(mymathapp PRIVATE ${CMAKE_SOURCE_DIR}/include)
            
      # 将库文件，可执行文件，头文件安装到指定目录
      install(TARGETS mymath mymathapp
              EXPORT MyMathTargets
              LIBRARY DESTINATION lib
              ARCHIVE DESTINATION lib
              RUNTIME DESTINATION bin
              PUBLIC_HEADER DESTINATION include
      )
      
    LIBRARY, ARCHIVE, RUNTIME, PUBLIC_HEADER是可选的，可以根据需要进行选择。 DESTINATION后面的路径可以自行制定，
    根目录默认为CMAKE_INSTALL_PREFIX,可以试用set方法进行指定，如果使用默认值的话，Unix系统的默认值为 /usr/local, 
    Windows的默认值为 c:/Program Files/${PROJECT_NAME}。比如字linux系统下若LIBRARY的安装路径指定为lib,即为/usr/local/lib
      
      
    他人使用我们编写的函数库，希望通过find_package方法进行引用，我们需要生成一个MyMathConfigVersion.cmake的文件来声明版本信息
      
      # 写入库的版本信息
      include(CMakePackageConfigHelpers)
      write_basic_package_version_file(
        MyMathConfigVersion.cmake
        VERSION ${PACKAGE_VERSION}
        COMPATIBILITY AnyNewerVersion  # 表示该函数库向下兼容
      )
      
    其中PACKAGE_VERSION 便是我们在CMakeLists.txt开头project(Installation VERSION 1.0)中声明的版本号
    
    第二步我们将前面 EXPORT MyMathTargets的信息写入到MyMathTargets.cmake文件中，
    该文件存放目录为${CMAKE_INSTALL_PREFIX}/lib/cmake/MyMath
    
      install(EXPORT MyMathTargets
        FILE MyMathTargets.cmake
        NAMESPACE MyMath::
        DESTINATION lib/cmake/MyLib
      )
      
    最后我们在源代码目录新建一个MyMathConfig.cmake.in文件，用于获取配置过程中的变量，并寻找项目依赖包。
      
      include(CMakeFindDependencyMacro)
      
      # 如果想要获取Config阶段的变量，可以使用这个
      # set(my-config-var @my-config-var@)
      
      # 如果你的项目需要依赖其他的库，可以使用下面语句，用法与find_package相同
      # find_dependency(MYDEP REQUIRED)
      
      include("${CMAKE_CURRENT_LIST_DIR}/MyMathTargets.cmake")
      
   最后在CMakeLists.txt文件中，配置生成MyMathTargets.cmake文件，并一同安装到${CMAKE_INSTALL_PREFIX}/lib/cmake/MyMath目录中
   
      configure_file(MyMathConfig.cmake.in MyMathConfig.cmake @ONLY)
      install(FILES "${CMAKE_CURRENT_BINARY_DIR}/MyMathConfig.cmake"
         "${CMAKE_CURRENT_BINARY_DIR}/MyMathConfigVersion.cmake"
          DESTINATION lib/cmake/MyMath
      )
        
   最后我们在其他项目中的CMakeLists.txt，就可以使用
   
    find_package(MyMath 1.0)
    target_linked_libraries(otherapp MyMath::mymath)
     
   来引用我们的函数库了。
   
      





                  
        
        
        
        
        
        
        
        
        
        
