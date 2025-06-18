# 目录
- [一、OpenCV的主要模块及核心简介](#一OpenCV的主要模块及核心简介)
- [二、Core模块(Core)](#二Core模块Core)
  - [1 核心数据结构](#1-核心数据结构)
    - [1.1 cv::Mat图像与多维矩阵的核心](#11-cvMat图像与多维矩阵的核心)
    - [1.2 cv::Point,cv::Size,cv::Rect,cv::Scalar](#12-cvPointcvSizecvRectcvScalar)
  - [2 基本操作与属性](#2-基本操作与属性)
    - [2.1 图像信息](#21-图像信息)
    - [2.2 类型转换](#22-类型转换)
    - [2.3 通道差分与合并](#23-通道差分与合并)
    - [2.4 数据归一化/缩放](#24-数据归一化缩放)
  - [3 绘制图像(常用于调试和可视化)](#3-绘制图像常用于调试和可视化)
  - [4 数学函数和数组运算](#4-数学函数和数组运算)
    - [4.1 通用矩阵运算](#41-通用矩阵运算)
    - [4.2 统计分析](#42-统计分析)
    - [4.3 矩阵操作](#43-矩阵操作)
  - [5 线性代数支持](#5-线性代数支持)
  - [6 常用的辅助函数](#6-常用的辅助函数)
- [三、图形处理模块(imgproc)](#三图形处理模块imgproc)
  - [1 图像滤波器](#1-图像滤波器)
    - [1.1 平滑/去噪滤波器](#11-平滑去噪滤波器)
      - [1.1.1 均值滤波Mean](#111-均值滤波Mean)
      - [1.1.2 高斯滤波GaussianBlur](#112-高斯滤波GaussianBlur)
      - [1.1.3 中值滤波Median](#113-中值滤波Median)
      - [1.1.4 双边滤波Bilateral](#114-双边滤波Bilateral)
      - [1.1.5 自定义核滤波](#115-自定义核滤波)
    - [1.2 锐化滤波器](#12-锐化滤波器)
      - [1.2.1 拉普拉斯Laplacian](#121-拉普拉斯Laplacian)
      - [1.2.2 Sobel](#122-Sobel)
      - [1.2.3 Scharr](#123-Scharr)
      - [1.2.4 锐化核](#124-锐化核)
    - [1.3 边缘检测滤波器](#13-边缘检测滤波器)
      - [1.3.1 Canny边缘检测](#131-Canny边缘检测)
      - [1.3.2 Sobel/Scharr](#132-SobelScharr)
      - [1.3.3 Laplacian](#133-Laplacian)
      - [1.3.4 Prewitt](#134-Prewitt)
      - [1.3.5 Roberts](#135-Roberts)
    - [1.4 频域滤波](#14-频域滤波)
      - [1.4.1 低通滤波器](#141-低通滤波器)
      - [1.4.2 高通滤波器](#142-高通滤波器)
      - [1.4.3 带通/带阻滤波器](#143-带通带阻滤波器)
    - [1.5 方向滤波/特殊滤波器](#15-方向滤波特殊滤波器)
      - [1.5.1 方向增强滤波器](#151-方向增强滤波器)
      - [1.5.2 Gabor滤波器](#152-Gabor滤波器)
      - [1.5.3 DoG 高斯差分](#153-DoG高斯差分)
  - [2 边缘检测](#2-边缘检测)
  - [3 几何变换](#3-几何变换)
    - [3.1 基本仿射变换](#31-基本仿射变换)
    - [3.2 投影/透视变换](#32-投影透视变换)
    - [3.3 重映射](#33-重映射)
    - [3.4 几何辅助函数](#34-几何辅助函数)
  - [4 形态学操作](#4-形态学操作)
    - [4.1 膨胀](#41-膨胀)
    - [4.2 腐蚀](#42-腐蚀)
    - [4.3 开运算](#43-开运算)
    - [4.4 闭运算](#44-闭运算)
    - [4.5 形态学梯度](#45-形态学梯度)
    - [4.6 顶帽](#46-顶帽)
    - [4.7 黑帽](#47-黑帽)
    - [4.8 击中击不中](#48-击中击不中)
  - [5 阈值化与直方图处理](#5-阈值化与直方图处理)
    - [5.1 阈值化](#51-阈值化)
    - [5.2 直方图处理](#52-直方图处理)
- [四、图像输入输出模块(imgcodecs)](#四图像输入输出模块imgcodecs)
- [五、视频I/O模块(videoio)](#五视频IO模块videoio)

# 一、OpenCV的主要模块及核心简介
- [x] Core模块(Core)
  - 作用：OpenCV的核心模块，提供基本的数据结构(如Mat)、数据操作、绘图函数等。
  - 内容:
    - cv::Mat图像矩阵数据结构
    - 向量、标量运算
    - 随机数生成器
    - 文件存储与读取(YAML，XML)
    - 线性代数与基本数学函数 
- [x] 图形处理模块(imgproc)
  - 作用: 图像的基本处理操作模块。
  - 内容：
    - 滤波器(平滑、锐化等)
    - 边缘检测(Canny、Sobel)
    - 几何变换(resize、warpAffine、warpPerspective)
    - 形态学操作(膨胀、腐蚀等)
    - 阈值化、直方图处理
- [x] 图像输入输出模块(imgcodecs)
  - 作用: 图像读写
  - 内容：
    - imread、imwrite、imdecode等
    - 支持JPEG、PNG、TIFF、BMP等  
- [x] 视频I/O模块(videoio)
  - 作用: 视频读取、写入以及摄像头访问
  - 内容:
    - VideoCapture、VideoWriter
    - 编码器、解码器 
- [ ] 对象检测与跟踪模块(objdetect)
  - 作用: 传统目标检测
  - 内容:
    - Haar特征分类器(人脸检测)
    - HOG(行人检测)
    - QR/条码检测    
- [ ] 机器学习模块(ml)
  - 作用：集成传统的ml算法
  - 内容：
    - SVM、KNN、决策树、随机森林等
    - 支持训练、预测和保存模型 
- [ ] 深度学习模块(dnn)
  - 作用：加载并运行DNN模型
  - 内容：
    - 支持ONNX/Caffe/TensorFlow等模型
    - 推理执行与blob数据封装
    - 常用于Yolo/ResNet/MobileNet等模型 
- [ ] 特征提取与描述子模块(features2d)
  - 作用: 关键点检测与描述
  - 内容：
    - SIFT、SURF(非自由)、ORB、BRISK等
    - 特征匹配、绘图、FLANN匹配等 
- [ ] 结构光与3D重建模块(calib3d)
  - 作用：相机标定与三维重建
  - 内容：
    - 单目/双目相机标定
    - 三维坐标重建
    - 立体匹配、深度图、投影变换 
- [ ] 运动分析与光流(video)
  - 作用：视频帧间分析
  - 内容：
    - 背景建模(MOG2)等
    - 光流估计(Lucas-Kannade、Farneback)
    - 多目标跟踪 
- [ ] GUI模块(highgui)
  - 作用：用于图像/视频的显示和简单的UI交互
  - 内容：
    - imshow、waitKey、createTrackbar 
- [ ] 扩展模块(contrib)
  - 作用：社区贡献模块，包含一些实验性功能和额外算法
  - 内容：
    - ArUco标记识别
    - Text字符检测与识别(OCR)
    - Face模块(人脸标记)

# 二、Core模块(Core)
## 1 核心数据结构
### 1.1 cv::Mat图像与多维矩阵的核心
- **自动内存管理**
```C++
Mat A = (Mat_<double>(6,6)<<
        1,1,1,1,1,1,
        2,2,2,2,2,2,
        3,3,3,3,3,3,
        4,4,4,4,4,4,
        5,5,5,5,5,5,
        6,6,6,6,6,6);  // 创建矩阵
cout<<"A:"<<A<<endl;
Mat B = A;  // 给A 起一个别名B，不复制任何数据
cout<<"B:"<<B<<endl;
Mat C = B.row(3);  // 为A 的第三行创建另一个标题；也不复制任何数据
cout<<"C:"<<C<<endl;
Mat D = B.clone();  // 创建单独的副本。
cout<<"D:"<<D<<endl;
B.row(5).copyTo(C);  // 将B的第5行复制到C,也就是A的第5行到第3行。
cout<<"C:"<<C<<endl;
cout<<"A:"<<A<<endl;
A = D;  // 现在让A和D共享数据；之后修改的版本，A仍然被B和C引用。
cout<<"A:"<<A<<endl;
B.release();  // 现在B成为一个空矩阵，不引用任何内存缓冲区，但修改后的A版本仍将被C引用，尽然C只是原始A的一行。
C = C.clone();  // 最后，制作C的完整副本。因此修改后的矩阵将被释放，因为它没有被任何引用。
cout<<"C:"<<C<<endl;
```
- **创建**
```C++
// 默认构造函数
Mat() CV_NOEXCEPT;

// rows:2d数组的行高
// cols:2d数组的列宽
// type:类型,CV8UC1,...,CV_64FC(n)
Mat(int rows, int cols, int type);

// size:2d数组 Size{cols,rows}
Mat(Size size, int type);

// s:初始化每个元素的值
Mat(int rows, int cols, int type, const Scalar& s);

Mat(Size size, int type, const Scalar& s);

// ndims: 数组维度
// sizes: 指定n维数组形状的整数数组
Mat(int ndims, const int* sizes, int type);

// sizes
Mat(const std::vector<int>& sizes, int type);

Mat(int ndims, const int* sizes, int type, const Scalar& s);

Mat(const std::vector<int>& sizes, int type, const Scalar& s);

// m：copy构造函数。不会复制任何数据，引用计数器(如果有的话)会增加。当修改
// 次矩阵，同时原矩阵的值也会修改。如果想要子数组的独立副本,请使用Mat::clone()
Mat(const Mat& m);

// data:指向用户数据的指针，没有数据复制，此操作非常高效，可用于使用OpenCV函数处理外部数据。外部数据不会自动释放。
// step:每个矩阵行占用的字节数。默认为AUTO_STEP，不假定填充，实际步长计算为cols*elemSize()。详见Mat::elemSize
Mat(int rows, int cols, int type, void* data, size_t step=AUTO_STEP);

Mat(Size size, int type, void* data, size_t step=AUTO_STEP);

// steps:在多维数组的情况下，ndims-1个步骤的数组(最后一步始终设置为元素大小)。如无指定，则假设矩阵是连续的。
Mat(int ndims, const int* sizes, int type, void* data, const size_t* steps=0);

Mat(const std::vector<int>& sizes, int type, void* data, const size_t* steps=0);

// m:不会复制任何数据，指向引用的数据。
// rowRange:要选取的m行的范围。左闭右开，使用Range::all()获取所有行
// colRange:要选取的m列的范围。左闭右开，使用Range::all()获取所有列
Mat(const Mat& m, const Range& rowRange, const Range& colRange=Range::all());

// m:不会复制任何数据，指向引用的数据。
// roi:指向指定区域
Mat(const Mat& m, const Rect& roi);

// m:不会复制任何数据，指向引用的数据。
// ranges: 每个维度选择的范围
Mat(const Mat& m, const Range* ranges);

Mat(const Mat& m, const std::vector<Range>& ranges);
......
```
- **复制/引用**
    - *clone()*：深拷贝
    - *copyTo()*: 复制到目标
    - *operator=*: 浅拷贝(共享数据)
- **元素访问**
  - *.at<T>(y,x)*: 类型安全访问
  - *ptr<T>(row)*: 行指针,效率高
  - *ptr<T>(y,x)*: 元素指针，效率高
- **子矩阵访问**
```
Mat roi = mat(Rect(x,y,w,h));
```
### 1.2 cv::Point,cv::Size,cv::Rect,cv::Scalar
- Point(x,y): 坐标
- Size(w,h): 宽高
- Rect(x,y,w,h): 矩形区域(常用于ROI)
- Scalar(b,g,r): 颜色值/多通道常数
## 2 基本操作与属性
### 2.1 图像信息
```
// 新的行矩阵底层数据与原始矩阵共享 以下均遵循OpenCV浅copy机制，除非特殊说明 时间复杂度O(1)
Mat row(int y) const;

// 时间复杂度O(1)
Mat row(int y) const;

// 时间复杂度O(1)
Mat rowRange(int startrow, int endrow) const;

Mat rowRange(const Range& r) const;

Mat colRange(int startcol, int endcol) const;

Mat colRange(const Range& r) const;

// 对角矩阵
Mat diag(int d=0) const;

// 完全数据的拷贝,即深拷贝
Mat clone() const;

// m: 目标矩阵，深拷贝
void copyTo(OutputArray m) const;

// mask: 与此大小相同的操作掩码，掩码必须为CV_8U类型，可以有一个或多个通道
void copyTo(OutputArray m, InputArray mask) const;

// 使用可选缩放将数组转换成另一种数据类型
// m: 输出矩阵
// rtype: 所需要的输出矩阵类型，更确切地说，深度，因为通道数量与输入相同；如果rtype为负,所需的输出矩阵将与输入具有相同的类型
// alpha: 可选比例因子，a>1:拉伸像素范围，增加对比度(亮部更亮，暗部更暗)。 0<a<1: 压缩像素范围，降低对比度(图像变灰)。a=1:图像不变。a<0:反转亮度(负片效果)，同时缩放对比度
// beta: 添加到缩放值中的可选增量。控制整体图片的亮度。 b>0: 增加亮度(所有像素值整体上移)。 b<0: 降低亮度(所有像素值整体下移)。
void convertTo(OutputArray m, int rtype, double alpha=1,double beta=0 ) const;

// 浅copy,底层共享数据，可转换数据类型，较少使用
void assignTo( Mat& m, int type=-1 ) const;

// 将输入的数组，设置成指定的值
// value: 分配的标量转换成实际的数据类型
// mask: 与此大小相同的操作掩码，非0的表示要复制的值，类型必须为CV_8U型，可以1或多通道
Mat& setTo(InputArray value, InputArray mask=noArray());

// 在不复制数据的情况下，改变2维矩阵的形状和通道数,创建一个新的矩阵头，时间复杂度O(1)
// cn: 新的通道数，如果为0，则与之前保持一致
// row: 新的行，如果为0，则与之前保持一致
Mat reshape(int cn, int rows=0) const;
 
// newndims: 新的维度数
// newsz: 新的全维矩阵大小，如果某些尺寸为0，则与之前保持一致
Mat reshape(int cn, int newndims, const int* newsz) const;

// newshape: 新的全维矩阵的vector，如果某些尺寸为0，则与之前保持一致
Mat reshape(int cn, const std::vector<int>& newshape) const;

// 转置矩阵，返回一个转置表达式，不执行实际的转置，可进一步做更复杂的矩阵表达式
MatExpr t() const;

// 逆矩阵，返回一个反转表达式，不执行实际的反转
MatExpr inv(int method=DECOMP_LU) const;

// 执行俩个矩阵的乘法
MatExpr mul(InputArray m, double scale=1) const;

// 计算两个3元向量的叉积。 应用场景3D场景，俩个平面的法向量
Mat cross(InputArray m) const;

// 计算俩个向量的点积
double dot(InputArray m) const;

// 返回指定大小和类型的零数组
// 返回一个Matlab风格的零数组初始化器。可以用来快速形成一个常量数组。
CV_NODISCARD_STD static MatExpr zeros(int rows, int cols, int type);
CV_NODISCARD_STD static MatExpr zeros(Size size, int type);
CV_NODISCARD_STD static MatExpr zeros(int ndims, const int* sz, int type);

// 返回指定大小和类型的所有1的数组
// 返回一个Matlab风格1的数组初始化器
CV_NODISCARD_STD static MatExpr ones(int rows, int cols, int type);
CV_NODISCARD_STD static MatExpr ones(Size size, int type);
CV_NODISCARD_STD static MatExpr ones(int ndims, const int* sz, int type);

// 返回指定大小和类型的单位矩阵
CV_NODISCARD_STD static MatExpr eye(int rows, int cols, int type);
CV_NODISCARD_STD static MatExpr eye(Size size, int type);

// 返回矩阵元素大小(以字节为单位)
// 通道数 * sizeof(类型)
size_t elemSize() const;

// 返回矩阵元素通道的大小(以字节为单位)
// sizeof(类型)
size_t elemSize1() const;

// 类型
int type() const;

// 位深度
int depth() const;

// 通道数
int channels() const;

是否为空
bool empty() const;

// 矩阵的所有元素大小
size_t total() const;

// 返回矩阵的行指针
template<typename _Tp> _Tp* ptr(int i0=0);

// 返回矩阵的元素指针
template<typename _Tp> _Tp* ptr(int row, int col);

// 元素访问的安全版本
template<typename _Tp> _Tp& at(int row, int col);
```
### 2.2 类型转换
```
// 线性变化 y=ax+b; 如下表示的是 a=1.0/255.0 b=0
mat.convertTo(dst,CV_32F,1.0/255.0); // 归一化
```
### 2.3 通道差分与合并
```
CV_EXPORTS_W void split(InputArray m, OutputArrayOfArrays mv);
CV_EXPORTS_W void merge(InputArrayOfArrays mv, OutputArray dst);
```
### 2.4 数据归一化/缩放
```
// 数据归一化，特征缩放，预处理
// 我想让所有像素乘以2再加10 -> convertTo (速度通常更快，静态，参数明确)
// 我想让图像像素最小值为0，最大值为255 -> normalize (复杂处理步骤多，动态，依赖输入数据的特征)
cv::normalize(src,dst,0,255,cv::NORM_MINMAX);

// 重置图像大小
// src: 原图像
// dst: 目标图像
// size: cv::Size指定大小
// fx: x的缩放因子
// fy: y的缩放因子
cv::resize(src,dst,size,fx,fy);
```
## 3 绘制图像(常用于调试和可视化)
```
// 线条
cv::line(img,p1,p2,color,thickness);

// 矩形区域
cv::rectangle(img,rect,color,thickness);

// 圆圈
cv::circle(img,center,radius,color,thickness);

绘制文本
cv::putText(img,text,pos,FONT_HERSHEY_SIMPLEX,scale,color);
```
## 4 数学函数和数组运算
### 4.1 通用矩阵运算
```
// 矩阵相加
// 输入矩阵与输出矩阵都可以具有相同或不同的深度。
// 类型大的兼容小的，可以通过dtype来指定大的类型
// 如果 src1.depth() == src2.depth()， 则默认 dtype=-1
CV_EXPORTS_W void add(InputArray src1, InputArray src2, OutputArray dst,InputArray mask = noArray(), int dtype = -1);

// 矩阵相减
// dst = src1 - src2  等同于 subtract(src1, src2 ,dst);
// dst -= src1 等同于 subtract(dst, src1, dst);
CV_EXPORTS_W void subtract(InputArray src1, InputArray src2, OutputArray dst,InputArray mask = noArray(), int dtype = -1);

// 矩阵相乘
// 计算俩个数组的按元素缩放的乘积
CV_EXPORTS_W void multiply(InputArray src1, InputArray src2,OutputArray dst, double scale = 1, int dtype = -1);

// 矩阵相除
// 执行每个元素的除法
CV_EXPORTS_W void divide(InputArray src1, InputArray src2, OutputArray dst, double scale = 1, int dtype = -1);

// 转换成半精度的float
CV_EXPORTS_W void convertFp16(InputArray src, OutputArray dst);

// 计算俩个数组的加权和，每个通道都是独立处理的
// dst = src1 * alpha + src2 * gamma;
CV_EXPORTS_W void addWeighted(InputArray src1, double alpha, InputArray src2, double beta, double gamma, OutputArray dst, int dtype = -1);

// 执行数组的查找表转换
// 是一种通过预定义映射关系快速修改像素值的技术，常用于：颜色风格化（如滤镜效果） 对比度增强 色彩空间压缩 医学图像伪彩色
CV_EXPORTS_W void LUT(InputArray src, InputArray lut, OutputArray dst);

// 计算俩个数组之间或数组和标量之间的每个元素的绝对差
// 非饱和运算，主要用于 差异检测、运动分析
CV_EXPORTS_W void absdiff(InputArray src1, InputArray src2, OutputArray dst);

// 计算俩个数组每个元素逻辑 &
// dst(i) = src1(i) & src2(i)
// 掩膜（Mask）应用	提取图像中感兴趣区域（ROI）
// 图像裁剪	结合矩形/圆形掩膜裁剪特定形状区域
// 位平面分解	分离图像的特定位平面（如最高有效位）
// 颜色过滤	通过阈值生成掩膜后提取特定颜色区域
CV_EXPORTS_W void bitwise_and(InputArray src1, InputArray src2, OutputArray dst, InputArray mask = noArray());

// 计算俩个数组每个元素逻辑 |
// dst(i) = src1(i) | src2(i)
// 图像合成	合并两个图像的ROI（如logo叠加）
// 多掩膜合并	将多个二值掩膜合并为一个
// 特征增强	增强图像中的特定特征（如边缘+角点组合）
// 加密水印	将水印信息嵌入到图像的低位平面
CV_EXPORTS_W void bitwise_or(InputArray src1, InputArray src2, OutputArray dst, InputArray mask = noArray());

// 计算俩个数组每个元素逻辑 ^
// dst(i) = src1(i) ^ src2(2)
// 相同为0，不同为1：0⊕0=0, 0⊕1=1, 1⊕0=1, 1⊕1=0
// 自反性：a ⊕ b ⊕ b = a（可用于数据加密/解密）
// 图像加密与数字水印 实现简单，解密快速
// 突出两幅图像的差异区域（比absdiff更敏感）
// 交互式绘图工具中切换像素选中状态。切换状态无需条件判断
// 快速校验数据完整性（如CRC校验的简化版）
CV_EXPORTS_W void bitwise_xor(InputArray src1, InputArray src2, OutputArray dst, InputArray mask = noArray());

```
### 4.2 统计分析
```
// 计算数组元素的平均值，每个通道单独计算并存储在Scalar中
CV_EXPORTS_W Scalar mean(InputArray src, InputArray mask = noArray());

// 计算数组元素的平均值和标准偏差
CV_EXPORTS_W void meanStdDev(InputArray src, OutputArray mean, OutputArray stddev, InputArray mask=noArray());

// 找到数组中全局的最大最小值，不适合多通道阵列，多通道的需要先通过reshape转换成单通道的
CV_EXPORTS_W void minMaxLoc(InputArray src, CV_OUT double* minVal,
                          CV_OUT double* maxVal = 0, CV_OUT Point* minLoc = 0,
                          CV_OUT Point* maxLoc = 0, InputArray mask = noArray());

// 找到数组中全局的最大与最小的索引
CV_EXPORTS void minMaxIdx(InputArray src, double* minVal, double* maxVal = 0,
                        int* minIdx = 0, int* maxIdx = 0, InputArray mask = noArray());

// 统计非零数组元素
CV_EXPORTS_W int countNonZero( InputArray src );
```
### 4.3 矩阵操作
```
// 围绕垂直 水平或俩个轴翻转二维码矩阵
// flipCode: 0 x-axis  1 y-axis  -1 xy-axis
CV_EXPORTS_W void flip(InputArray src, OutputArray dst, int flipCode);

// 垂直拼接
CV_EXPORTS void hconcat(InputArray src1, InputArray src2, OutputArray dst);

// 水平拼接
CV_EXPORTS void vconcat(InputArray src1, InputArray src2, OutputArray dst);

// 转置矩阵
CV_EXPORTS_W void transpose(InputArray src, OutputArray dst);

// 重复填充,用输入数组的重复副本填充输出数组
// ny: src沿垂直方向重复的次数
// nx: src沿水平方向重复的次数
CV_EXPORTS_W void repeat(InputArray src, int ny, int nx, OutputArray dst);
```
## 5 线性代数支持
```
MatExpr inv(int method=DECOMP_LU) const;

MatExpr t() const;

// 行列式
// mtx: 类型必须具有CV_32FC1或CV_64FC1类型和平方大小的输入矩阵
CV_EXPORTS_W double determinant(InputArray mtx);

// 解一个或多个线性系统或最小二乘问题
// 函数cv::solve解决线性系统或最小二乘问题
CV_EXPORTS_W bool solve(InputArray src1, InputArray src2, OutputArray dst, int flags = DECOMP_LU);

// 计算对称矩阵的特征值和特征向量
CV_EXPORTS_W bool eigen(InputArray src, OutputArray eigenvalues, OutputArray eigenvectors = noArray());
```
## 6 常用的辅助函数
```
// 多线程与opencv内核设置
cv::setNumThreads(4);
int threads = cv::getNumThreads();
cv::useOptimized(true);

// 调试辅助函数
cv::namedWindow("debug")
cv::imshow("debug",mat)
cv::waitKey(0);
cv::destroyAllWindows();
cv::imwrite("debug.jpg",mat);

// 随机生成图像
// 生成一个均匀分布的随机数或随机数数组
// dst: 随机数输出数组，必须预先分配
// low: 生成随机数的包含下限
// high: 生成随机数的排他性上限
CV_EXPORTS_W void randu(InputOutputArray dst, InputArray low, InputArray high);
// 用正态分布随机填充数组
// dst: 随机数输出数组，必须预先分配，并且具有1~4个信道
// mean: 生成的随机数的平均值(期望值)
// stddev: 生成随机数的标准偏差，可以是向量
CV_EXPORTS_W void randn(InputOutputArray dst, InputArray mean, InputArray stddev);

// 高精度计时
CV_EXPORTS_W int64 getTickCount();
CV_EXPORTS_W double getTickFrequency();

```
# 三、图形处理模块(imgproc)
## 1 图像滤波器
|任务|推荐滤波器|原因|
|:--:|:--:|:--:|
|金属表面划痕|高斯滤波+Laplacian|去噪+边缘增强|
|胶囊缺口检测|中值滤波+Canny|去噪+边缘提取清晰|
|背景建模(光照变化)|高斯滤波/双边滤波|平滑不均匀背景|
|OCR提取|自适应高斯+Sobel|背景压制+字符突出|
|激光线提取|Gabor/Sobel(垂直)|精准提取单方向结构|
|实时视频降噪|中值滤波(高速)/时域滤波|快速去除噪点，实时处理|
### 1.1 平滑/去噪滤波器
#### 1.1.1 均值滤波Mean
均值滤波是一种最常见的图像平滑(模糊)技术，它的基本原理是通过一个卷积核(通常是一个大小为奇数的矩阵)对图像进行处理，
将每个像素点的值替换为其领域(窗口)像素值的平均值。它的主要作用是去除图像中的噪声，同时保持图像的大致结构，但可能会模糊一些边缘信息。
**CV_EXPORTS_W void blur( InputArray src, OutputArray dst, Size ksize, Point anchor = Point(-1,-1), int borderType = BORDER_DEFAULT );**     
- ksize: 卷积核大小,通常是一个奇数，3,5,7。 数值越大，平滑效果越强，图像的细节会丢失更多。
- anchor: 锚点，(-1,-1)表示位于内核中心，Point(0,0)表示卷积核的左上角对齐到当前像素位置。
- borderType: 边界处理方式，处理图像边界时的策略，决定了滤波器如何处理图像边缘的像素。  
  - BORDER_CONSTANT: 常数填充，边界区域填充为指定常数值
  - BORDER_REPLICATE：复制边界像素
  - BORDER_REFLECT：反射边界像素
  - BORDER_WRAP：环绕边界像素
  - BORDER_REFLECT_101：反射边界像素，去除边界像素

// 更灵活的均值滤波实现，允许用户指定卷积核的大小和边界条件     
CV_EXPORTS_W void boxFilter( InputArray src, OutputArray dst, int ddepth,
                           Size ksize, Point anchor = Point(-1,-1),
                           bool normalize = true,
                           int borderType = BORDER_DEFAULT );    

// 自定义卷积核      
CV_EXPORTS_W void filter2D( InputArray src, OutputArray dst, int ddepth,
                          InputArray kernel, Point anchor = Point(-1,-1),
                          double delta = 0, int borderType = BORDER_DEFAULT );

#### 1.1.2 高斯滤波GaussianBlur
**CV_EXPORTS_W void GaussianBlur( InputArray src, OutputArray dst, Size ksize,double sigmaX, double sigmaY = 0, int borderType = BORDER_DEFAULT );**    
使用高斯滤波器对图像进行卷积。支持**就地滤波**    
- src: 输入图像，可任意通道数，每个通道独立处理，深度应为CV_8U, CV_16U, CV_16S, CV_32F, CV_64F    
- dst: 输出图像，与src大小和类型相同    
- ksize: 高斯核大小。ksize.width和ksize.height可以不同，但必须为正奇数。或者，它们可以为零，然后根据sigma计算得出    
- sigmaX：高斯核在X方向上的标准差    
- sigmaY: 高斯核在Y方向上的标准差，如果simgaY为零，则将其设置为等于simgaX;如果俩个都为零，则分别根据ksize.width和ksize.height计算得出    
- borderType: 详见BorderTypes,目前运行效果看，应该是线条的宽度

|*sigmaX*|*模糊程度*|*图像表现*|
|:--:|:--:|:--:|
|0|不模糊(无效果)|图像保持原样|
|~0.3|非常轻微|几乎无变化|
|1.0|中度模糊|一般模糊，部分边缘保留|
|2.0|强模糊|细节严重丢失，边缘柔化|
|5.0+|极强模糊|类似高斯光晕，结构消失|

工业场景建议
|*应用场景*|*推荐sigmaX*|
|:--:|:--:|
|表面噪声去除|0.5~1.0|
|背景建模模糊|1.0~2.5|
|高斯差分检测|0.5~2.0|
|光照不均处理前预平滑|2.0+|
#### 1.1.3 中值滤波Median
中值滤波是一种非线性滤波技术，常用于去噪、平滑图像。
其主要特点是通过取领域内像素的中值来替代当前像素值。与均值滤波不同，均值滤波是对领域内的像素值进行加权平均
而中值滤波通过排序选择领域的中值，这样可以有效去除图像中的“椒盐噪声”
**CV_EXPORTS_W void medianBlur( InputArray src, OutputArray dst, int ksize );**    
- src: 输入图像
- dst: 输出图像(经过中值滤波后的结果)
- ksize: 滤波器的大小，必须是奇数(3,5,7等),较大的值产生更强的平滑效果，但也会模糊图像细节
应用场景：特别适合去除椒盐噪声，边缘保持，修复损坏图像

#### 1.1.4 双边滤波Bilateral
双边滤波是一种非线性滤波技术，在平滑图像的同时，能够较好地保留图像的边缘细节。相较于其他滤波器，它非常地慢     
核心思想：不仅考虑空间距离(像素位置的距离)，还考虑像素值的相似性。
- 空间距离权重：领域内像素的空间距离越近，权重越大
- 像素值差异权重：像素值差异越小，权重越大     
**CV_EXPORTS_W void bilateralFilter( InputArray src, OutputArray dst, int d,**
                                 **double sigmaColor, double sigmaSpace,**
                                 **int borderType = BORDER_DEFAULT );**
- d: 滤波过程中使用的每个像素邻域的直径。如果它不是正的，则根据sigmaSpace计算。 通常设置5-15之间，d=9是较为通用的值，适用于大多数图像
- sigmaColor: 颜色空间的标准差，决定了像素值相似性的影响程度。较大的sigmaColor使得像素值差异较大的区域也会受到影响。通常75能达到较好的效果
- sigmaSpace: 空间坐标的标准差，决定了空间距离的影响范围。较大的sigmaSpace会让较远的像素也参与计算。 通常保持在50~100,75通常能达到较好的效果
- borderType: borderType——用于推断图像外部像素的边框模式，请参阅#BorderTypes。
应用场景：去噪且保边缘，图像美化与去斑点，边缘检测前的平滑(如Canny,Sobel)，去除高斯噪声与斑点噪声，动漫风格处理。

|d领域直径|sigmaColor(颜色标准差)|sigmaSpace(空间标准差)|备注|
|:--:|:--:|:--:|:--:|
|5|30~50|30~50|适用于细节较多的图像，效果较锐利|
|9|50~100|50~100|通用设置，适用于大部分图像|
|15|100~150|100~150|平滑效果更强，适合较大图像|

#### 1.1.5 自定义核滤波
**CV_EXPORTS_W void filter2D( InputArray src, OutputArray dst, int ddepth,**
                          **InputArray kernel, Point anchor = Point(-1,-1),**
                          **double delta = 0, int borderType = BORDER_DEFAULT );**     
kernel卷积核
- 均值滤波
```
Mat kernel=(Mat_<float>(3,3) << 1/9.0,1/9.0,1/9.0,
                                1/9.0,1/9.0,1/9.0,
                               1/9.0,1/9.0,1/9.0);
```
- 锐化滤波
```
Mat kernel=(Mat_<float>(3,3) << 0,-1,0,
                               -1,5,-1,
                                0,-1,0);
```
- 边缘检测(Sobel核)
```
Mat kernel=(Mat_<float>(3,3) << -1,0,1,
                                -2,0,2,
                                -1,0,1);
```
自定义卷积核的设计在于如何选择权重。不同的权重组合会产生不同的图像效果。
一些常见的自定义卷积核设计思路:
- 锐化：通过在中心位置增加更大的权重，周围值为负数，使图像的细节变得更加清晰。
- 平衡/模糊: 通过平均领域的像素值来去除图像的噪声或细节，使图像看起来更平滑。
- 边缘检测：使用Sobel、Prewitt、Laplacian等边缘检测核，通过加强图像的梯度信息来提取图像边缘。
- 降噪：设计核进行平滑，从而去除图像中噪声
### 1.2 锐化滤波器
锐化的本质是强调图像中像素变化较大的区域(通常是边缘)，其原理通常是基于拉普拉斯算子或高通滤波：
- 强化边缘(像素变化剧烈的地方)
- 抑制平滑区域(像素差异较小)
#### 1.2.1 拉普拉斯Laplacian
Laplacian是一种二阶导数运算符
```
0  1  0
1 -4  1
0  1  0

1  1  1
1 -8  1
1  1  1 
```
**CV_EXPORTS_W void Laplacian( InputArray src, OutputArray dst, int ddepth,**
                           **int ksize = 1, double scale = 1, double delta = 0,**
                           **int borderType = BORDER_DEFAULT );**     
应用场景：
- 边缘检测：检测图像中边缘区域(轮廓、目标边界等)，比一阶导数(如Sobel)检测更强的变化
- 图像锐化：使用Laplacian算子后将其结果加回原图(增强边缘)，图像增强、照片清晰化、OCR前处理
```
Mat sharpended = src - 0.7 * dst;
```
- 特征提取前预处理：目标检测、关键点匹配、SLAM等
- 图像分割辅助：强化边界区域，使得分割算法更加精确
#### 1.2.2 Sobel 
Sobel算子是一种边缘检测算法，通过对图像来做梯度计算，来检测图像中的边缘信息。     
// src: 原图
// dst: 目标图
// ddepth: 图像深度
// dx: 导数x的阶数
// dy: 导数y的阶数
// ksize: 核函数，必须是1,3,5,7
// scale: 计算导数值的可选比例因子
// delta: 偏移量
// borderType: 边界类型
CV_EXPORTS_W void Sobel( InputArray src, OutputArray dst, int ddepth,
                         int dx, int dy, int ksize = 3,
                         double scale = 1, double delta = 0,
                         int borderType = BORDER_DEFAULT );     
Sobel算子使用俩个3x3的卷积核分别计算
```
x方向的梯度(Gx)
[-1 0 1]
[-2 0 2]
[-1 0 1]
y方向的梯度(Gy)
[-1  -2  -1]
[ 0   0   0]
[ 1   2   1]
```
应用场景：
- 图像边缘检测：较Canny更简单，适合快速检测大致轮廓
- 车道线检测：用于检测水平或垂直方向变化明显的线段
- 图像梯度提取：可用于纹理方向分析
使用建议：
- 输入图建议先GaussianBlur降噪，再进行Sobel,以获得更干净的边缘。
- 想要获得更清晰的效果，可以尝试Scharr算子，是sobel的优化版，尤其适合ksize=3的场景

#### 1.2.3 Scharr
Scharr算子是一种用于图像边缘检测的一阶导数滤波器，设计目标是提供比Sobel更好的旋转对称性和抗噪性能，特别是3x3，可提供比Sobel更准确的梯度估计。    
Scharr卷积核
```
Gx:
[-3  0  3]
[-10 0  10]
[-3  0  3]

Gy:
[-3  -10  -3]
[ 0   0    0]
[ 3   10   3]
```
与Sobel的最大权2不同，Scharr的最大权10，提升中心像素的权重，从而在边缘检测中获得更精确的梯度估计。

// src: 原图
// dst: 目标图像
// ddepth: 深度
// dx: 可导x的阶数
// dy: 可导y的阶数
// scale: 缩放比率
// delta: 偏移量
// borderType: 边界类型
**CV_EXPORTS_W void Scharr( InputArray src, OutputArray dst, int ddepth,**
                          **int dx, int dy, double scale = 1, double delta = 0,**
                          **int borderType = BORDER_DEFAULT );**     
应用场景：
- 精细边缘提取，对边缘、物体边界要求高的检测
- 工业视觉检测，检测边缘是否破损、微小裂缝
- OCR预处理，比较清晰的边界有助于字符分割与识别
- 动态视频帧的边缘提取，对实时噪声鲁棒
小技巧：
- 输出深度建议CV_16S或CV_32F，再用convertScaleAbs转为CV_8U可视化
- cv::addWeighted 可以调整横/竖方向边缘的权重，突出某一方向。
- 结合GaussianBlur预处理效果更好

#### 1.2.4 锐化核
锐化操作的核心思想是：**增强图像中灰度变换剧烈的区域(即边缘),抑制平衡区域的灰度值**     
常见的锐化卷积核
```
// 标准锐化核(常用)
[ 0  -1   0]
[-1   5  -1]
[ 0  -1   0]

// 强化版锐化核(增强对比)
[-1  -1  -1]
[-1   9  -1]
[-1  -1  -1]

// 自定义锐化(柔和锐化)
[ 1  -2   1]
[-2   9  -2]
[ 1  -2   1]
```
锐化注意事项:
- 噪声敏感，锐化会放大图像中的噪声，建议搭配降噪滤波(如高斯)使用
- 溢出风险，锐化操作可能导致像素值越界，需正确设置输出深度
- 边缘模糊图片，在较模糊图像中锐化更明显，但也易引入伪边缘

### 1.3 边缘检测滤波器
#### 1.3.1 Canny边缘检测
Canny边缘检测算法是一种多阶段的边缘检测方法，具有抗噪性强、定位精度高的特点。     
处理流程：
- 降噪处理
- 梯度计算
- 非极大值抑制(NMS)
- 双阈值处理
- 边缘连接

// image: 单通道图像
// edges: 与image保持相同
// threshold1: 第一个阈值
// threshold2: 第二个阈值
// apertureSize: Sobel算子的大小
// L2gradient: 标识符，是否启用更加精确的梯度计算方式，默认是简化版
**CV_EXPORTS_W void Canny( InputArray image, OutputArray edges,**
                         **double threshold1, double threshold2,**
                         **int apertureSize = 3, bool L2gradient = false );**

|阈值组合|效果说明|
|:--:|:--:|
|50  150|常用默认设置，适合大多数图像|
|100 200|对噪声敏感度较低，边缘清晰|
|10  60|弱边缘也能保留，噪声较多时慎用|
|自适应阈值|可以通过cv::mean()或cv::adaptiveThreshold()来自动设置阈值|
一般建议先高斯滤波，然后使用经验或自动方式设置阈值

应用场景：
- 图像轮廓提取，提取图像中的物体边缘，构建边界框，轮廓
- 文档分析，提取文字区域边缘，帮助分割文本和背景
- 机器人视觉，边缘信息用于目标识别与路径规划
- 医学图像分析，提取病灶、骨骼、血管等结构边界
- 游戏开发，做特殊风格边缘描边特效
- 摄像头动态分析，实时边缘流数据，用于目标检测前处理
- 工业检测，检测零件轮廓是否标准，有无毛刺、缺陷等
#### 1.3.2 Sobel/Scharr
[Sobel](#122-Sobel)    
[Scharr](#123-Scharr)
#### 1.3.3 Laplacian 
[Laplacian](#121-拉普拉斯-Laplacian)
#### 1.3.4 Prewitt
Prewitt算子是一阶导数边缘检测方法，与Sobel相似，但权重较均匀(不加权中心点)，它用来计算图像的水平和垂直梯度。
```
水平梯度Gx:
[-1 0 1]
[-1 0 1]
[-1 0 1]

垂直梯度Gy:
[ 1  1  1]
[ 0  0  0]
[-1 -1 -1]
```
应用场景：
- 普通图像边缘检测，精度适中，噪声鲁棒性中
- 工业缺陷检测，对结构边缘较明显的图像适用
- 教学，常用于边缘检测教学和比较
#### 1.3.5 Roberts
Roberts交叉梯度算子是最早的边缘检测算法，使用2x2的卷积核，适合检测锐利边缘，但抗噪能力差
```
水平梯度Gx:
[1  0]
[0 -1]

垂直梯度Gy:
[ 0 1]
[-1 0]
```
应用场景:
- 医学图像，用于轮廓锐利但图像分辨率较低的医学图像
- 数码图像分析，处理早期数字图像，硬件资源受限场景
- 老旧系统，算法轻量级，适合嵌入式或早期图像处理平台

**Prewitt vs Roberts vs Sobel 对比**
|算子|卷积核大小|抗噪声能力|边缘精度|计算速度|应用稳定性|
|:--:|:--:|:--:|:--:|:--:|:--:|
|Prewitt|3x3|中|中|中|较稳定|
|Roberts|2x2|弱|高(锐)|快速|不稳定(易受噪声影响)|
|Sobel|3x3|强|中|中|最常用|

### 1.4 频域滤波
#### 1.4.1 低通滤波器
图像可看作是空间频率的组合：    
- 高频：图像中的边缘、纹理、噪点等，表现为亮度或颜色的快速变化。
- 低频：图像中变化缓慢的区域，如平坦背景或渐变区域。
低通滤波器就是允许低频通过，抑制高频

常见的低通滤波器类型(空间域)

|类型|特点与说明|OpenCV函数|
|:--:|:--:|:--:|
|均值滤波器|简单平均周围像素|cv::blur|
|高斯滤波器|根据高斯分布加权平均,更平滑|cv::GaussianBlur|
|中值滤波器|中间值替代中心像素，抗噪效果好|cv::medianBlur|
|双边滤波|同时考虑空间距离与像素相似性|cv::bilateralFilter|

应用场景：
- 图像降噪 去除感光器产生的高频噪声
- 模糊处理 实现图像虚化、磨皮、柔焦
- 图像金字塔 多分辨率图像构建前的预处理
- 模版匹配 降低匹配对高频纹理的敏感度
- 特征提取前 去除小细节、稳定边缘检测

#### 1.4.2 高通滤波器
高通滤波器是图像处理中的一种重要滤波器，主要用于保存图像中的高频信息(如边缘、细节)并抑制低频信息(如平滑区域)      
高通滤波器的目标：增强高频信息(细节、轮廓)抑制低频信息(模糊、背景)     
原理简介:
- 高通滤波器通常将高频保留(值为1),将低频抑制(值为0或较小)
- 可以通过傅里叶变换实现频域处理，也可以用空域卷积方式实现近似效果

常见实现方式(空域):
- [锐化滤波器](#124-锐化核)
- [拉普拉斯增强](#121-拉普拉斯Laplacian)

应用场景：
- 边缘检测 强化边缘，做轮廓提取前的准备，与Canny/Sobel算子配合使用
- 图像增强 提高图像对比度，细节突出，卷积锐化
- 图像分割 保留纹理特征用于区域分割，在医学/工业图像中尤为有用
- 图像复原 高通反卷积以恢复模糊图像(逆运动模糊) 在图像恢复、模糊去除
- 计算机视觉 特征提取做预处理 人脸识别、物体检测等

#### 1.4.3 带通/带阻滤波器
带通滤波器：只保留某一特定频率范围的图像成分(通常介于低频与高频之间)     
带阻滤波器：抑制某一特定频率范围，保留其它低频/高频部分

应用场景：
- 材料纹理分析 提取某种特定频率纹理特征
- 模式识别/缺陷检测 屏蔽某种频率带的干扰或特征
- 去周期噪声 带阻滤波器可抑制图像中周期性条纹或干扰波
- 视觉增强 保留对比度显著的纹理区域，去除大面积低频背景
```
Mat img = imread("1.jpg",IMREAD_GRAYSCALE);
// 扩展图像尺寸为2的幂大小(优化DFT效率)
int m = getOptimalDFTSize(img.rows);
int n = getOptimalDFTSize(img.cols);
Mat padded;
// 边缘扩充函数，在图像的四周增加像素边界(上下左右)，并按指定的方式填充这些新区域。
copyMakeBorder(img,padded,0,m-img.rows,0,n-img.cols,BORDER_CONSTANT,cv::Scalar::all(0));  
// 傅里叶变换(DFT)
Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(),CV_32F)};
Mat complexImg;
merge(planes, 2, complexImg);
dft(complexImg, complexImg);
// 构造带通 / 带阻掩膜
int cx = complexImg.cols / 2;
int cy = complexImg.rows / 2;
float r_inner = 30.0f;  // 内圈半径
float r_outer = 60.0f;  // 外圈半径

cv::Mat mask = cv::Mat::zeros(complexImg.size(), CV_32F);

for (int y = 0; y < mask.rows; ++y) {
    for (int x = 0; x < mask.cols; ++x) {
        float dist = std::sqrt(std::pow(x - cx, 2) + std::pow(y - cy, 2));
        // 带通：只保留一定范围
        if (dist >= r_inner && dist <= r_outer) {
            mask.at<float>(y, x) = 1.0f;
        }
    }
}

// 创建2通道掩码
cv::Mat mask2ch[] = {mask.clone(), mask.clone()};
cv::Mat complexMask;
cv::merge(mask2ch, 2, complexMask);
// 应用掩膜
mulSpectrums(complexImg, complexMask, complexImg, 0);
// 反变换还原图像
cv::Mat invDFT;
cv::idft(complexImg, invDFT, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
cv::Mat finalImg;
invDFT(cv::Rect(0, 0, img.cols, img.rows)).convertTo(finalImg, CV_8U);
```

### 1.5 方向滤波/特殊滤波器
#### 1.5.1 方向增强滤波器 
- Sobel滤波器(方向性边缘检测)
```
cv::Mat sobelX, sobelY
cv::Sobel(src, sobelX,CV_32F,1,0);  // 1,0 表示在X方向取一阶导数
cv::Sobel(src, sobelY,CV_32F,0,1);  // 0,1 表示在Y方向取一阶导数
```
- Sharr滤波器(更精确的方向导数)
```
cv::Mat sharrX,scharrY;
cv::Scharr(src, scharrX,CV_32F,1,0); // x方向
cv::Scharr(src, scharrY,CV_32F,0,1); // y方向
// 相较于Sobel, Scharr在小核尺寸下效果更好，尤其在边缘检测时更准确。
```
#### 1.5.2 Gabor滤波器
用于特定方向/频率增强    
```
// CV_PI/4 表示滤波器的方向为45°
// 可通过改变方向参数设计多个方向的滤波器，提取不同方向纹理
cv::Mat kernel = cv::getGaborKernel(cv::Size(21,21),4.0,CV_PI/4,10.0,0.5,0,CV_32F);
cv::Mat filtered;
cv::filter2D(src,filtered,CV_32F,kernel);
```
#### 1.5.3 DoG高斯差分
DoG(Difference of Gaussians) = 俩个不同尺度的高斯模糊图像之间的差值    
DoG近似于拉普拉斯高斯(LoG)
```
Mat src = imread("input.jpg",IMREAD_GRAYSCALE);
Mat gauss1,gauss2,dog;
double sigma1 = 1.0;
double sigma2 = 2.0;
GaussianBlur(src,gauss1,Size(0,0),sigma1);
GaussianBlur(src,gauss2,Size(0,0),sigma2);
// 相减得到DoG图像
subtract(gauss1,gauss2,dog,noArray(),CV_32F);
normalize(dog,dog,0,255,NORM_MINMAX);
dog.convertTo(dog,CV_8U);
```
## 2 边缘检测
常用于:
- 物体检测与分割
- 图像特征提取(如轮廓、角点)
- 图像增强
边缘检测的基本流程:
1. **预处理**: 图像灰度化 + 去噪(高斯模糊)
2. **梯度计算**: 检测像素灰度的突变(常用Sobel、Laplacian、Canny)
3. **非极大值抑制**: 某些算法
4. **阈值处理**: 决定是否保留某条边缘
常用边缘检测方法：
1. **Canny**: 经典高性能算法 最常用
```
Mat img = imread("input.jpg",IMREAD_GRAYSCALE);
Mat blurred, edges;
GaussianBlur(img,blurred,Size(5,5),1.4);
Canny(blurred,edges,100,200);  // 低,高阈值
// 优点：边缘细，连续性强
// 参数调整关键：俩个阈值(低高阈值用于边缘连接)
```
2. **Sobel**: 一阶导数，计算梯度方向
```
Mat grad_x,grad_y,abs_x,abs_y,grad;
Sobel(img,grad_x,CV_16S,1,0);
Sobel(img,grad_y,CV_16S,0,1);
convertScaleAbs(grad_x,abs_x);
convertScaleAbs(grad_y,abs_y);
addWeighted(abs_x,0.5,abs_y,0.5,0,grad);
// 用于检测水平方向和垂直方向的边缘
// 可以进一步求出梯度角度(用于方向判断)
```
3. **Laplacian**: 二阶导数，检测强烈边缘
```
Mat lap,laplacian;
Laplacian(img,lap,CV_16S,3);
convertScaleAbs(lap,laplacian);
// 敏感度高，容易噪声大
// 可用于检测图像快速变化的区域
```
4. **Scharr**: 比Sobel更精细的梯度算子
```
Mat gray_x,gray_y,grad;
Scharr(img,grad_x,CV_16S,1,0);
Scharr(img,grad_y,CV_16S,0,1);
// 精度比Sobel高，但用途类似
```
示例：边缘检测 + 轮廓提取
```
Mat img = imread("input.jpg",IMREAD_GRAYSCALE);
Mat edges;
Canny(img,edges,100,200);
vector<vector<Point>> contours;
findContours(edges,contours,RETR_EXTERNAL,CHAIN_APPROX_SIMPLE);
Mat result = Mat::zeros(img.size(),CV_8UC3);
drawContours(result,contours,-1,Scalar(0,255,0));
```
## 3 几何变换
### 3.1 基本仿射变换
1. **缩放**: cv::resize
```
// 重置图片大小
// 如果仅想调整图片大小  resize(src, dst, dst.size(), 0, 0, interpolation)
// 如果想放大图片为原来2倍  resize(src, dst, Size(), 0.5, 0.5 ,interpolation)
// 缩小图片，INTER_AREA 插值效果好
// 放大图片，INTER_CUBIC(慢)或INTER_LINEAR(更快些)
CV_EXPORTS_W void resize( InputArray src, OutputArray dst,
                          Size dsize, double fx = 0, double fy = 0,
                          int interpolation = INTER_LINEAR );
```
2. **旋转**: cv::getRotationMatrix2D + cv::warpAffine
```
// 计算二维旋转的仿射矩阵
// center: 源图像中的旋转中心
// angle: 旋转角度，单位为度, 正值表示逆时针旋转（坐标原点假定为左上角）
// scale: 各向同性标度因子
CV_EXPORTS_W Mat getRotationMatrix2D(Point2f center, double angle, double scale);

// 对图像应用的仿射变换
// M: 2x3的变换矩阵
CV_EXPORTS_W void warpAffine( InputArray src, OutputArray dst,
                              InputArray M, Size dsize,
                              int flags = INTER_LINEAR,
                              int borderMode = BORDER_CONSTANT,
                              const Scalar& borderValue = Scalar());
```
3. **平移**： 直接构造仿射矩阵
```
Mat trans = (Mat_<double>(2,3) << 1,0,tx,0,1,ty);
warpAffine(src, dst, trans, src.size());
```
4. **任意仿射变换(三点对应)**: cv::getAffineTransform
```
vector<Point2f> srcTri = {p1,p2,p3};
vector<Point2f> dstTri = {q1,q2,q3};
Mat M = getAffineTransform(srcTri, dstTri);
warpAffine(src, dst, M, src.size());
```
### 3.2 投影/透视变换
特点：支持"远大近小"效果(透视投影)     
适用于：车道线矫正、二维码识别、图像配准、投影矫正等。
```
vector<Point2f> srcPts{...}, dstPts{...};
Mat M = getPerspectiveTransform(srcPts, dstPts);
warpPerspective(src, dst, M, dstSize);

// 从4对点计算透视变换
// solveMethod: 详见 enum DecompTypes
CV_EXPORTS_W Mat getPerspectiveTransform(InputArray src, InputArray dst, int solveMethod = DECOMP_LU);
```
### 3.3 重映射
适合自定义每个像素怎么映射(如鱼眼纠正、图像扭曲)
```
Mat map_x,map_y;
remap(src,dst,map_x,map_y,INTER_LINEAR);

// 对图像进行通用几何变换
CV_EXPORTS_W void remap( InputArray src, OutputArray dst,
                         InputArray map1, InputArray map2,
                         int interpolation, int borderMode = BORDER_CONSTANT,
                         const Scalar& borderValue = Scalar());
```
### 3.4 几何辅助函数
插值方法
|插值方式|枚举值|说明|
|:--:|:--:|:--:|
|最近邻|INTER_NEAREST|快速但不平滑|
|双线性|INTER_LINEAR|默认值，平滑|
|三次卷积|INTER_CUBIC|效果更好,计算慢|
|Lanczos|INTER_LANCZOS4|高清图像时效果好|

## 4 形态学操作
主要用于: 
- 去噪
- 连接物体
- 提取边界
- 分割图像结构
本质是基于“结构元素(kernel)”对图像的像素进行处理，特别适合二值图像(黑白图),但也可用于灰度图    
**核心工具**：结构元素
```
Mat kernel = getStructuringElement(MORPH_RECT,Size(3,3));
```
- 矩形: MORPH_REACT
- 椭圆: MORPH_ELLIPSE
- 十字: MORPH_CROSS
### 4.1 膨胀
```
Mat kernel = getStructuringElement(MORPH_RECT,Size(3,3))
Mat src = imread("input.jpg",IMREAD_GRAYSCALE)
Mat dilated;
// 白的区域会变大
dilate(src, dilated,kernel);
```
### 4.2 腐蚀
```
Mat kernel = getStructuringElement(MORPH_RECT,Size(3,3))
Mat src = imread("input.jpg",IMREAD_GRAYSCALE)
Mat eroded;
// 白的区域会变小
erode(src, dilated,kernel);
```
### 4.3 开运算
```
// 等于 erode -> dilate
// 用于去除小颗粒噪声，保留大物体
cv::morphologyEx(src, opened, cv::MORPH_OPEN, kernel);
```
### 4.4 闭运算
```
// 等于 dilate -> erode
// 用于连接断裂的白色区域或填补黑点
cv::morphologyEx(src, opened, cv::MORPH_CLOSE, kernel);
```
### 4.5 形态学梯度
```
// 等于 dialte - erode
// 图像边缘最明显，适合找轮廓
cv::morphologyEx(src, grad, cv::MORPH_GRADIENT, kernel);
```
### 4.6 顶帽
```
cv::morphologyEx(src, tophat, cv::MORPH_TOPHAT, kernel);     // 原图 - 开  突出亮细节  在图像增强与照明补偿中有用
```
### 4.7 黑帽
```
cv::morphologyEx(src, blackhat, cv::MORPH_BLACKHAT, kernel); // 闭 - 原图 突出暗细节  在图像增强与照明补偿中有用
```
### 4.8 击中击不中
```
// 用于形状匹配检测 只能用于二值图（0/1），不常用
cv::Mat thres;
cv::threshold(src, thres, 127, 1, cv::THRESH_BINARY); // 图像需为0/1值
cv::morphologyEx(thres, hitmiss, cv::MORPH_HITMISS, kernel);  
```
汇总
| 操作名称  | OpenCV 函数                            | 用途       |
| ----- | ------------------------------------ | -------- |
| 膨胀    | `cv::dilate`                         | 扩大白色区域   |
| 腐蚀    | `cv::erode`                          | 收缩白色区域   |
| 开运算   | `cv::morphologyEx(..., MORPH_OPEN)`  | 去小白点     |
| 闭运算   | `cv::morphologyEx(..., MORPH_CLOSE)` | 连续白色块、填洞 |
| 梯度    | `MORPH_GRADIENT`                     | 提取边缘     |
| 顶帽    | `MORPH_TOPHAT`                       | 提亮细节     |
| 黑帽    | `MORPH_BLACKHAT`                     | 提亮暗部     |
| 击中击不中 | `MORPH_HITMISS`                      | 二值图形态匹配  |

## 5 阈值化与直方图处理
### 5.1 阈值化
```
// THRESH_BINARY	大于阈值为 maxval，否则为 0	白底黑字
// THRESH_BINARY_INV	小于阈值为 maxval，否则为 0	黑底白字
// THRESH_TRUNC	大于阈值则设为阈值，其它保持	裁剪过亮
// THRESH_TOZERO	小于阈值设为 0，大于保持原值	保留高亮
// THRESH_TOZERO_INV	大于阈值设为 0，小于保持	保留暗部
double cv::threshold(
    InputArray src,        // 输入图像（8-bit 单通道）
    OutputArray dst,       // 输出图像
    double thresh,         // 阈值
    double maxval,         // 最大值
    int type               // 阈值类型
);

cv::Mat gray = cv::imread("img.jpg", cv::IMREAD_GRAYSCALE);
cv::Mat binary;
cv::threshold(gray, binary, 127, 255, cv::THRESH_BINARY);

// 自适应阈值(局部不同)
// blockSize: 局部区域大小(必须是奇数)
// C: 从局部均值中减去的常数
cv::adaptiveThreshold(src, dst, 255,
                      cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                      cv::THRESH_BINARY,
                      blockSize, C);

// Otsu 自动阈值法
// 自动选出最佳全局阈值（图像必须是单通道灰度图）
cv::threshold(src, dst, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
```
### 5.2 直方图处理
- 计算直方图
```
// image：输入图像（灰度/彩色都行）
// channels：通道索引（灰度用 {0}）
// hist：输出直方图
// histSize：多少个灰度 bin（如 256）
// ranges：灰度范围 [0, 256]
cv::calcHist(&image, 1, channels, cv::Mat(), hist, 1, &histSize, &ranges);
```
- 绘制直方图(伪图)
```
int hist_w = 512, hist_h = 400;
int bin_w = cvRound((double) hist_w / histSize);

cv::Mat histImage(hist_h, hist_w, CV_8UC1, cv::Scalar(0));

cv::normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX);

for (int i = 1; i < histSize; i++) {
    cv::line(histImage,
             cv::Point(bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1))),
             cv::Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
             cv::Scalar(255), 2);
}
```
- 直方图均衡化（增强对比）
```
// 必须8位单通道灰度图
// 结果图像对比度增强，细节更清晰
cv::equalizeHist(gray, equalized);
```
# 四、图像输入输出模块(imgcodecs)
1. cv::imread
```
// cv::IMREAD_COLOR 默认 加载彩色图
// cv::IMREAD_GRAYSCALE 加载灰度图
// cv::IMRead_UNCHANGED 保留alpha通道
cv::Mat img = cv::imread("",cv::IMREAD_COLOR);
```
2. cv::imwrite
```
cv::imwrite("result.png",img);

std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
cv::imwrite("result.jpg",img,params);
```
3. cv::imencode
```
// 将图像编码为内存缓存(std::vector<uchar>)
// 用于网络传输或数据库存储
std::vector<uchar> buf;
cv::imencode(".png",img,buf);
```
4. cv::imdecode
```
// 从内存缓冲解码为cv::Mat
// 用于网络接受到字节流还原图像
cv::Mat img = cv::imdecode(buf,cv::IMREAD_COLOR);
```
# 五、视频I/O模块(videoio)
videoio是OpenCV中用于视频流和相机IO的模块，作用：
- 打开视频文件或相机(视频输入)
- 从视频流中逐帧读取图像
- 创建视频文件(视频输出)
- 编码、解码视频帧
- 支持多种后端(如FFmpeg,DirectShow,GStreamer,V4L,MSMF等)

1. cv::VideoCapture
```
cv::VideoCapture cap(0);  // 打开默认相机

cv::VideoCapture cap("video.mp4");  // 打开指定的视频
```

|方法|功能|
|:--:|:--:|
|read(Mat&)|读取一帧|
|grab()|抓取下一帧(不解码)|
|retrieve(Mat&)|获取抓取的帧|
|get(propId)|获取视频属性(如宽高、帧率)|
|set(propId,value)|设置属性|
```
常用属性(propId):
- CAP_PROP_FRAME_WIDTH：帧宽
- CAP_PROP_FRAME_HEIGHT：帧高
- CAP_PROP_FPS：帧率
- CAP_PROP_POS_FRAMES：当前帧编号
- CAP_PROP_FRAME_COUNT：总帧数
```

2. cv::VideoWriter
```
cv::VideoCapture cap(0);
if(!cap.isOpened()){
  std::cerr << "无法打开相机" << std::endl;
  return -1;
}
cv::VideoWriter writer("out.avi",cv::VideoWriter::fourcc('M','J','P','G'),30,cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH),(int)cap.get(cv::CAP_PROP_FRAME_HEIGHT)));
if (!writer.isOpened()){
  std::cerr << "无法打开视频写入器" << std::endl;
  return -1;
}
cv::Mat frame;
while (true){
  cap >> frame;  // 等价于cap.read(frame)
  if (frame.empty()) break;
  writer.write(frame);
  cv::imshow("Camera",frame);
  if (cv::waitKey(30)==27) break;
}
```
