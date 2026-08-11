# Python Extension 例子
## 单文件无依赖
> 1. 编写cpp文件
```cpp
#include <vector>
#include <random>
#include <algorithm>
#include <string> // 新增：引入 string 头文件
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::vector<std::vector<int>> generate_grid(int width, int height, const std::string& seed_str) {
    const std::vector<int> all_angles = {0, 90, 180, 270};
    std::vector<std::vector<int>> grid(height, std::vector<int>(width, -1));

    std::seed_seq seed(seed_str.begin(), seed_str.end());
    std::mt19937 gen(seed);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            std::vector<int> available = all_angles;
        
            if (y > 0) {
                int up_angle = grid[y-1][x];
                available.erase(std::remove(available.begin(), available.end(), up_angle), available.end());
            }

            if (x > 0) {
                int left_angle = grid[y][x-1];
                available.erase(std::remove(available.begin(), available.end(), left_angle), available.end());
            }

            std::uniform_int_distribution<> dis(0, available.size() - 1);
            grid[y][x] = available[dis(gen)];
        }
    }

    return grid;
}

// 绑定代码
PYBIND11_MODULE(graph_color, m) {
    m.doc() = "Graph coloring random grid generator"; 
    
    // 绑定时增加 seed_str 参数，并给一个默认值（例如 "default_seed"）
    m.def("generate_grid", &generate_grid, "Generate a 2D grid with no adjacent identical angles",
          py::arg("width") = 10, py::arg("height") = 10, py::arg("seed_str") = "default_seed");
}
```
> 2. 下载python依赖库
```shell
pip install setuptools pybind11
```
> 3. 构建自动化脚本
```python
# 文件名 setup.py 自定义
from setuptools import setup 
from pybind11.setup_helpers import Pybind11Extension, build_ext

# 定义扩展模块: (模块名，[源文件列表])
ext_modules = [
  Pybind11Extension("graph_color", ["graph_coloring.cpp"])
]

setup(
  name="graph_color",
  ext_modules=ext_modules,
  # 使用pybind11提供build_ext, 它会自动注入最高效的编译参数
  cmdclass={"build_ext": build_ext} 
)
```
> 4. 执行编译命令
```shell
// build_ext: 告诉setuptools去编译c/c++扩展
// --inplace: 原地编译，编译生成.pyd(动态库)
python setup.py build_ext --inplace
```

## 依赖第三方库
> 1. 编译setup.py 
```python
import os 
from setuptools import setup 
from pybind11.setup_helpers import Pybind11Extension, build_ext

# 1. 定义MSYS2 UCRT64下的第三方绝对路径
# 可以使用系统变量动态设置
MSYS2_UCRT64_PREFIX="D:/Msys64/ucrt64"

# OpenCV的头文件目录
opencv_include = os.path.join(MSYS2_UCRT64_PREFIX, "include", "opencv4")
ucrt64_lib = os.path.join(MSYS2_UCRT64_PREFIX, "lib")

ext_modules = [
  Pybind11Extension(
    name="image_processor",  # 生成python的模块名
    sources=["image_processor.cpp"],  # c++源文件
    include_dirs=[opencv_include],  # 头文件路径 -I
    library_dirs=[ucrt64_lib],  # 库文件搜索路径 -L 
    libraries=[
      "opencv_core",
      "opencv_imgproc",
      "opencv_highgui",
    ],
    # 可选 额外的编译参数
    extra_compile_args=["-O3","-Wall"],
  ),
]

setup(
  name="image_processor",
  ext_modules=ext_modules,
  cmdclass={"build_ext":build_ext},
)
```


