# VCPKG 分步骤安装opencv4

> * 了解一下vcpkg安装目录
>> ports: 该文件夹存放库的编译选项以及一些配置，很重要，里面每个库都会有相对于的文件夹，比如opencv4,里面portfile.cmake是最核心的修改配置   
>> buildtrees: 是存放库解压后的源码以及cmake对应cmake的一些文件，还有一些编译日志   
>> downloads: 是存放下载的库的压缩包 命名规则 opencv-opencv-4.7.0-1.tar.gz   规律是 库名-压缩包名, 因此可以自行下载之后放在该路径下进行安装库   
>> packages: 相当于缓存一样   
>> installed: 安装后的库以及头文件    
> * 下载opencv4
> `vcpkg install opencv4:wasm32-emscripten --only-downloads`
> * 修改配置 wasm不支持simd，需要禁用一些参数 文件在 ~/vcpkg/ports/opencv4/portfile.cmake   在函数 vcpkg_cmake_configure 下 OPTIONS 增加如下内容
> ```
> -DCMAKE_CXX_FLAGS="-mno-sse"
> -DCMAKE_C_FLAGS="-mno-sse"
> -DENABLE_PRECOMPILED_HEADERS=OFF
> -DBUILD_opencv_world=ON
> -DBUILD_opencv_python2=OFF
> -DBUILD_opencv_python3=OFF
> -DBUILD_SHAREED_LIBS=OFF
> -DENABLE_PIC=OFF
> -DBUILD_TESTS=OFF
> -DBUILD_PERF_TESTS=OFF
> -DBUILD_EXAMPLES=OFF
> -DWITH_IPP=OFF
> -DWITH_OPENXR=OFF
> -DBUILD_opencv_apps=OFF
> -DWITH_ITT=OFF
> -DCV_DISABLE_OPTIMIZATION=ON
> ```
> * vcpkg编译  
> `vcpkg install opencv4:wasm32-emscripten --no-downloads --editable`
