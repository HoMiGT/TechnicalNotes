# 目录
- [一、OpenCV的主要模块及核心简介](#一OpenCV的主要模块及核心简介)
- [二、Core模块(Core)](#二Core模块Core)
  - [1 核心数据结构](#1-核心数据结构)
    - [1.1 cv::Mat 图像与多维矩阵的核心](#11-cv::Mat-图像与多维矩阵的核心)
    - [1.2 cv::Point,cv::Size,cv::Rect,cv::Scalar](#12-cv::Point,cv::Size,cv::Rect,cv::Scalar)
  - [2 基本操作与属性](#2-基本操作与属性)
    - [2.1 图像信息](#21-图像信息)
    - [2.2 类型转换](#22-类型转换)
    - [2.3 通道差分与合并](#23-通道差分与合并)
    - [2.4 数据归一化/缩放](#24-数据归一化/缩放)
  - [3 绘制图像(常用于调试和可视化)](#3-绘制图像常用于调试和可视化)
  - [4 数学函数和数组运算](#4-数学函数和数组运算)
    - [4.1 通用矩阵运算](#41-通用矩阵运算)
    - [4.2 统计分析](#42-统计分析)
    - [4.3 矩阵操作](#43-矩阵操作)
  - [5 线性代数支持](#5-线性代数支持)
  - [6 常用的辅助函数](#6-常用的辅助函数)
# 一、OpenCV的主要模块及核心简介
> - [x] Core模块(Core)
>   - 作用：OpenCV的核心模块，提供基本的数据结构(如Mat)、数据操作、绘图函数等。
>   - 内容:
>     - cv::Mat图像矩阵数据结构
>     - 向量、标量运算
>     - 随机数生成器
>     - 文件存储与读取(YAML，XML)
>     - 线性代数与基本数学函数 
> - [ ] 图形处理模块(imgproc)
>   - 作用: 图像的基本处理操作模块。
>   - 内容：
>     - 滤波器(平滑、锐化等)
>     - 边缘检测(Canny、Sobel)
>     - 几何变换(resize、warpAffine、warpPerspective)
>     - 形态学操作(膨胀、腐蚀等)
>     - 阈值化、直方图处理
> - [ ] 图像输入输出模块(imgcodecs)
>   - 作用: 图像读写
>   - 内容：
>     - imread、imwrite、imdecode等
>     - 支持JPEG、PNG、TIFF、BMP等  
> - [ ] 视频I/O模块(videoio)
>   - 作用: 视频读取、写入以及摄像头访问
>   - 内容:
>     - VideoCapture、VideoWriter
>     - 编码器、解码器 
> - [ ] 对象检测与跟踪模块(objdetect)
>   - 作用: 传统目标检测
>   - 内容:
>     - Haar特征分类器(人脸检测)
>     - HOG(行人检测)
>     - QR/条码检测    
> - [ ] 机器学习模块(ml)
>   - 作用：集成传统的ml算法
>   - 内容：
>     - SVM、KNN、决策树、随机森林等
>     - 支持训练、预测和保存模型 
> - [ ] 深度学习模块(dnn)
>   - 作用：加载并运行DNN模型
>   - 内容：
>     - 支持ONNX/Caffe/TensorFlow等模型
>     - 推理执行与blob数据封装
>     - 常用于Yolo/ResNet/MobileNet等模型 
> - [ ] 特征提取与描述子模块(features2d)
>   - 作用: 关键点检测与描述
>   - 内容：
>     - SIFT、SURF(非自由)、ORB、BRISK等
>     - 特征匹配、绘图、FLANN匹配等 
> - [ ] 结构光与3D重建模块(calib3d)
>   - 作用：相机标定与三维重建
>   - 内容：
>     - 单目/双目相机标定
>     - 三维坐标重建
>     - 立体匹配、深度图、投影变换 
> - [ ] 运动分析与光流(video)
>   - 作用：视频帧间分析
>   - 内容：
>     - 背景建模(MOG2)等
>     - 光流估计(Lucas-Kannade、Farneback)
>     - 多目标跟踪 
> - [ ] GUI模块(highgui)
>   - 作用：用于图像/视频的显示和简单的UI交互
>   - 内容：
>     - imshow、waitKey、createTrackbar 
> - [ ] 扩展模块(contrib)
>   - 作用：社区贡献模块，包含一些实验性功能和额外算法
>   - 内容：
>     - ArUco标记识别
>     - Text字符检测与识别(OCR)
>     - Face模块(人脸标记)

# 二、Core模块(Core)
## 1 核心数据结构
### 1.1 cv::Mat 图像与多维矩阵的核心
> - **自动内存管理**
> ```C++
> Mat A = (Mat_<double>(6,6)<<
>         1,1,1,1,1,1,
>         2,2,2,2,2,2,
>         3,3,3,3,3,3,
>         4,4,4,4,4,4,
>         5,5,5,5,5,5,
>         6,6,6,6,6,6);  // 创建矩阵
> cout<<"A:"<<A<<endl;
> Mat B = A;  // 给A 起一个别名B，不复制任何数据
> cout<<"B:"<<B<<endl;
> Mat C = B.row(3);  // 为A 的第三行创建另一个标题；也不复制任何数据
> cout<<"C:"<<C<<endl;
> Mat D = B.clone();  // 创建单独的副本。
> cout<<"D:"<<D<<endl;
> B.row(5).copyTo(C);  // 将B的第5行复制到C,也就是A的第5行到第3行。
> cout<<"C:"<<C<<endl;
> cout<<"A:"<<A<<endl;
> A = D;  // 现在让A和D共享数据；之后修改的版本，A仍然被B和C引用。
> cout<<"A:"<<A<<endl;
> B.release();  // 现在B成为一个空矩阵，不引用任何内存缓冲区，但修改后的A版本仍将被C引用，尽然C只是原始A的一行。
> C = C.clone();  // 最后，制作C的完整副本。因此修改后的矩阵将被释放，因为它没有被任何引用。
> cout<<"C:"<<C<<endl;
> ```
> - **创建**
> ```C++
> // 默认构造函数
> Mat() CV_NOEXCEPT;
>
> // rows:2d数组的行高
> // cols:2d数组的列宽
> // type:类型,CV8UC1,...,CV_64FC(n)
> Mat(int rows, int cols, int type);
>
> // size:2d数组 Size{cols,rows}
> Mat(Size size, int type);
>
> // s:初始化每个元素的值
> Mat(int rows, int cols, int type, const Scalar& s);
>
> Mat(Size size, int type, const Scalar& s);
>
> // ndims: 数组维度
> // sizes: 指定n维数组形状的整数数组
> Mat(int ndims, const int* sizes, int type);
>
> // sizes
> Mat(const std::vector<int>& sizes, int type);
>
> Mat(int ndims, const int* sizes, int type, const Scalar& s);
>
> Mat(const std::vector<int>& sizes, int type, const Scalar& s);
>
> // m：copy构造函数。不会复制任何数据，引用计数器(如果有的话)会增加。当修改
> // 次矩阵，同时原矩阵的值也会修改。如果想要子数组的独立副本,请使用Mat::clone()
> Mat(const Mat& m);
>
> // data:指向用户数据的指针，没有数据复制，此操作非常高效，可用于使用OpenCV函数处理外部数据。外部数据不会自动释放。
> // step:每个矩阵行占用的字节数。默认为AUTO_STEP，不假定填充，实际步长计算为cols*elemSize()。详见Mat::elemSize
> Mat(int rows, int cols, int type, void* data, size_t step=AUTO_STEP);
>
> Mat(Size size, int type, void* data, size_t step=AUTO_STEP);
>
> // steps:在多维数组的情况下，ndims-1个步骤的数组(最后一步始终设置为元素大小)。如无指定，则假设矩阵是连续的。
> Mat(int ndims, const int* sizes, int type, void* data, const size_t* steps=0);
>
> Mat(const std::vector<int>& sizes, int type, void* data, const size_t* steps=0);
>
> // m:不会复制任何数据，指向引用的数据。
> // rowRange:要选取的m行的范围。左闭右开，使用Range::all()获取所有行
> // colRange:要选取的m列的范围。左闭右开，使用Range::all()获取所有列
> Mat(const Mat& m, const Range& rowRange, const Range& colRange=Range::all());
>
> // m:不会复制任何数据，指向引用的数据。
> // roi:指向指定区域
> Mat(const Mat& m, const Rect& roi);
>
> // m:不会复制任何数据，指向引用的数据。
> // ranges: 每个维度选择的范围
> Mat(const Mat& m, const Range* ranges);
>
> Mat(const Mat& m, const std::vector<Range>& ranges);
>
> ......
> ```
> - **复制/引用**
>   - *clone()*：深拷贝
>   - *copyTo()*: 复制到目标
>   - *operator=*: 浅拷贝(共享数据)
> - **元素访问**
>   - *.at<T>(y,x)*: 类型安全访问
>   - *ptr<T>(row)*: 行指针,效率高
>   - *ptr<T>(y,x)*: 元素指针，效率高
> - **子矩阵访问**
> ```
> Mat roi = mat(Rect(x,y,w,h));
> ```
### 1.2 cv::Point,cv::Size,cv::Rect,cv::Scalar
> - Point(x,y): 坐标
> - Size(w,h): 宽高
> - Rect(x,y,w,h): 矩形区域(常用于ROI)
> - Scalar(b,g,r): 颜色值/多通道常数
## 2 基本操作与属性
### 2.1 图像信息
> ```
> // 新的行矩阵底层数据与原始矩阵共享 以下均遵循OpenCV浅copy机制，除非特殊说明 时间复杂度O(1)
> Mat row(int y) const;
>
> // 时间复杂度O(1)
> Mat row(int y) const;
>
> // 时间复杂度O(1)
> Mat rowRange(int startrow, int endrow) const;
>
> Mat rowRange(const Range& r) const;
>
> Mat colRange(int startcol, int endcol) const;
>
> Mat colRange(const Range& r) const;
>
> // 对角矩阵
> Mat diag(int d=0) const;
>
> // 完全数据的拷贝,即深拷贝
> Mat clone() const;
>
> // m: 目标矩阵，深拷贝
> void copyTo(OutputArray m) const;
>
> // mask: 与此大小相同的操作掩码，掩码必须为CV_8U类型，可以有一个或多个通道
> void copyTo(OutputArray m, InputArray mask) const;
>
> // 使用可选缩放将数组转换成另一种数据类型
> // m: 输出矩阵
> // rtype: 所需要的输出矩阵类型，更确切地说，深度，因为通道数量与输入相同；如果rtype为负,所需的输出矩阵将与输入具有相同的类型
> // alpha: 可选比例因子，a>1:拉伸像素范围，增加对比度(亮部更亮，暗部更暗)。 0<a<1: 压缩像素范围，降低对比度(图像变灰)。a=1:图像不变。a<0:反转亮度(负片效果)，同时缩放对比度
> // beta: 添加到缩放值中的可选增量。控制整体图片的亮度。 b>0: 增加亮度(所有像素值整体上移)。 b<0: 降低亮度(所有像素值整体下移)。
> void convertTo(OutputArray m, int rtype, double alpha=1,double beta=0 ) const;
>
> // 浅copy,底层共享数据，可转换数据类型，较少使用
> void assignTo( Mat& m, int type=-1 ) const;
>
> // 将输入的数组，设置成指定的值
> // value: 分配的标量转换成实际的数据类型
> // mask: 与此大小相同的操作掩码，非0的表示要复制的值，类型必须为CV_8U型，可以1或多通道
> Mat& setTo(InputArray value, InputArray mask=noArray());
>
> // 在不复制数据的情况下，改变2维矩阵的形状和通道数,创建一个新的矩阵头，时间复杂度O(1)
> // cn: 新的通道数，如果为0，则与之前保持一致
> // row: 新的行，如果为0，则与之前保持一致
> Mat reshape(int cn, int rows=0) const;
> 
> // newndims: 新的维度数
> // newsz: 新的全维矩阵大小，如果某些尺寸为0，则与之前保持一致
> Mat reshape(int cn, int newndims, const int* newsz) const;
>
> // newshape: 新的全维矩阵的vector，如果某些尺寸为0，则与之前保持一致
> Mat reshape(int cn, const std::vector<int>& newshape) const;
>
> // 转置矩阵，返回一个转置表达式，不执行实际的转置，可进一步做更复杂的矩阵表达式
> MatExpr t() const;
>
> // 逆矩阵，返回一个反转表达式，不执行实际的反转
> MatExpr inv(int method=DECOMP_LU) const;
>
> // 执行俩个矩阵的乘法
> MatExpr mul(InputArray m, double scale=1) const;
>
> // 计算两个3元向量的叉积。 应用场景3D场景，俩个平面的法向量
> Mat cross(InputArray m) const;
>
> // 计算俩个向量的点积
> double dot(InputArray m) const;
>
> // 返回指定大小和类型的零数组
> // 返回一个Matlab风格的零数组初始化器。可以用来快速形成一个常量数组。
> CV_NODISCARD_STD static MatExpr zeros(int rows, int cols, int type);
> CV_NODISCARD_STD static MatExpr zeros(Size size, int type);
> CV_NODISCARD_STD static MatExpr zeros(int ndims, const int* sz, int type);
>
> // 返回指定大小和类型的所有1的数组
> // 返回一个Matlab风格1的数组初始化器
> CV_NODISCARD_STD static MatExpr ones(int rows, int cols, int type);
> CV_NODISCARD_STD static MatExpr ones(Size size, int type);
> CV_NODISCARD_STD static MatExpr ones(int ndims, const int* sz, int type);
>
> // 返回指定大小和类型的单位矩阵
> CV_NODISCARD_STD static MatExpr eye(int rows, int cols, int type);
> CV_NODISCARD_STD static MatExpr eye(Size size, int type);
>
> // 返回矩阵元素大小(以字节为单位)
> // 通道数 * sizeof(类型)
> size_t elemSize() const;
>
> // 返回矩阵元素通道的大小(以字节为单位)
> // sizeof(类型)
> size_t elemSize1() const;
>
> // 类型
> int type() const;
>
> // 位深度
> int depth() const;
>
> // 通道数
> int channels() const;
>
> 是否为空
> bool empty() const;
>
> // 矩阵的所有元素大小
> size_t total() const;
>
> // 返回矩阵的行指针
> template<typename _Tp> _Tp* ptr(int i0=0);
>
> // 返回矩阵的元素指针
> template<typename _Tp> _Tp* ptr(int row, int col);
>
> // 元素访问的安全版本
> template<typename _Tp> _Tp& at(int row, int col);
> ```
### 2.2 类型转换
> ```
> // 线性变化 y=ax+b; 如下表示的是 a=1.0/255.0 b=0
> mat.convertTo(dst,CV_32F,1.0/255.0); // 归一化
> ```
### 2.3 通道差分与合并
> ```
> CV_EXPORTS_W void split(InputArray m, OutputArrayOfArrays mv);
> CV_EXPORTS_W void merge(InputArrayOfArrays mv, OutputArray dst);
> ```
### 2.4 数据归一化/缩放
> ```
> // 数据归一化，特征缩放，预处理
> // 我想让所有像素乘以2再加10 -> convertTo (速度通常更快，静态，参数明确)
> // 我想让图像像素最小值为0，最大值为255 -> normalize (复杂处理步骤多，动态，依赖输入数据的特征)
> cv::normalize(src,dst,0,255,cv::NORM_MINMAX);
>
> // 重置图像大小
> // src: 原图像
> // dst: 目标图像
> // size: cv::Size指定大小
> // fx: x的缩放因子
> // fy: y的缩放因子
> cv::resize(src,dst,size,fx,fy);
> ```
## 3. 绘制图像(常用于调试和可视化)
> ```
> // 线条
> cv::line(img,p1,p2,color,thickness);
>
> // 矩形区域
> cv::rectangle(img,rect,color,thickness);
>
> // 圆圈
> cv::circle(img,center,radius,color,thickness);
>
> 绘制文本
> cv::putText(img,text,pos,FONT_HERSHEY_SIMPLEX,scale,color);
> ```
## 4. 数学函数和数组运算
### 4.1 通用矩阵运算
> ```
> // 矩阵相加
> // 输入矩阵与输出矩阵都可以具有相同或不同的深度。
> // 类型大的兼容小的，可以通过dtype来指定大的类型
> // 如果 src1.depth() == src2.depth()， 则默认 dtype=-1
> CV_EXPORTS_W void add(InputArray src1, InputArray src2, OutputArray dst,InputArray mask = noArray(), int dtype = -1);
>
> // 矩阵相减
> // dst = src1 - src2  等同于 subtract(src1, src2 ,dst);
> // dst -= src1 等同于 subtract(dst, src1, dst);
> CV_EXPORTS_W void subtract(InputArray src1, InputArray src2, OutputArray dst,InputArray mask = noArray(), int dtype = -1);
>
> // 矩阵相乘
> // 计算俩个数组的按元素缩放的乘积
> CV_EXPORTS_W void multiply(InputArray src1, InputArray src2,OutputArray dst, double scale = 1, int dtype = -1);
>
> // 矩阵相除
> // 执行每个元素的除法
> CV_EXPORTS_W void divide(InputArray src1, InputArray src2, OutputArray dst, double scale = 1, int dtype = -1);
>
> // 转换成半精度的float
> CV_EXPORTS_W void convertFp16(InputArray src, OutputArray dst);
>
> // 计算俩个数组的加权和，每个通道都是独立处理的
> // dst = src1 * alpha + src2 * gamma;
> CV_EXPORTS_W void addWeighted(InputArray src1, double alpha, InputArray src2, double beta, double gamma, OutputArray dst, int dtype = -1);
>
> // 执行数组的查找表转换
> // 是一种通过预定义映射关系快速修改像素值的技术，常用于：颜色风格化（如滤镜效果） 对比度增强 色彩空间压缩 医学图像伪彩色
> CV_EXPORTS_W void LUT(InputArray src, InputArray lut, OutputArray dst);
>
> // 计算俩个数组之间或数组和标量之间的每个元素的绝对差
> // 非饱和运算，主要用于 差异检测、运动分析
> CV_EXPORTS_W void absdiff(InputArray src1, InputArray src2, OutputArray dst);
>
> // 计算俩个数组每个元素逻辑 &
> // dst(i) = src1(i) & src2(i)
> // 掩膜（Mask）应用	提取图像中感兴趣区域（ROI）
> // 图像裁剪	结合矩形/圆形掩膜裁剪特定形状区域
> // 位平面分解	分离图像的特定位平面（如最高有效位）
> // 颜色过滤	通过阈值生成掩膜后提取特定颜色区域
> CV_EXPORTS_W void bitwise_and(InputArray src1, InputArray src2, OutputArray dst, InputArray mask = noArray());
>
> // 计算俩个数组每个元素逻辑 |
> // dst(i) = src1(i) | src2(i)
> // 图像合成	合并两个图像的ROI（如logo叠加）
> // 多掩膜合并	将多个二值掩膜合并为一个
> // 特征增强	增强图像中的特定特征（如边缘+角点组合）
> // 加密水印	将水印信息嵌入到图像的低位平面
> CV_EXPORTS_W void bitwise_or(InputArray src1, InputArray src2, OutputArray dst, InputArray mask = noArray());
>
> // 计算俩个数组每个元素逻辑 ^
> // dst(i) = src1(i) ^ src2(2)
> // 相同为0，不同为1：0⊕0=0, 0⊕1=1, 1⊕0=1, 1⊕1=0
> // 自反性：a ⊕ b ⊕ b = a（可用于数据加密/解密）
> // 图像加密与数字水印 实现简单，解密快速
> // 突出两幅图像的差异区域（比absdiff更敏感）
> // 交互式绘图工具中切换像素选中状态。切换状态无需条件判断
> // 快速校验数据完整性（如CRC校验的简化版）
> CV_EXPORTS_W void bitwise_xor(InputArray src1, InputArray src2, OutputArray dst, InputArray mask = noArray());
>
> ```
### 4.2 统计分析
> ```
> // 计算数组元素的平均值，每个通道单独计算并存储在Scalar中
> CV_EXPORTS_W Scalar mean(InputArray src, InputArray mask = noArray());
>
> // 计算数组元素的平均值和标准偏差
> CV_EXPORTS_W void meanStdDev(InputArray src, OutputArray mean, OutputArray stddev, InputArray mask=noArray());
>
> // 找到数组中全局的最大最小值，不适合多通道阵列，多通道的需要先通过reshape转换成单通道的
> CV_EXPORTS_W void minMaxLoc(InputArray src, CV_OUT double* minVal,
>                           CV_OUT double* maxVal = 0, CV_OUT Point* minLoc = 0,
>                           CV_OUT Point* maxLoc = 0, InputArray mask = noArray());
>
> // 找到数组中全局的最大与最小的索引
> CV_EXPORTS void minMaxIdx(InputArray src, double* minVal, double* maxVal = 0,
>                         int* minIdx = 0, int* maxIdx = 0, InputArray mask = noArray());
>
> // 统计非零数组元素
> CV_EXPORTS_W int countNonZero( InputArray src );
> ```
### 4.3 矩阵操作
> ```
> // 围绕垂直 水平或俩个轴翻转二维码矩阵
> // flipCode: 0 x-axis  1 y-axis  -1 xy-axis
> CV_EXPORTS_W void flip(InputArray src, OutputArray dst, int flipCode);
>
> // 垂直拼接
> CV_EXPORTS void hconcat(InputArray src1, InputArray src2, OutputArray dst);
>
> // 水平拼接
> CV_EXPORTS void vconcat(InputArray src1, InputArray src2, OutputArray dst);
>
> // 转置矩阵
> CV_EXPORTS_W void transpose(InputArray src, OutputArray dst);
>
> // 重复填充,用输入数组的重复副本填充输出数组
> // ny: src沿垂直方向重复的次数
> // nx: src沿水平方向重复的次数
> CV_EXPORTS_W void repeat(InputArray src, int ny, int nx, OutputArray dst);
> ```
## 5. 线性代数支持
> ```
> MatExpr inv(int method=DECOMP_LU) const;
>
> MatExpr t() const;
>
> // 行列式
> // mtx: 类型必须具有CV_32FC1或CV_64FC1类型和平方大小的输入矩阵
> CV_EXPORTS_W double determinant(InputArray mtx);
>
> // 解一个或多个线性系统或最小二乘问题
> // 函数cv::solve解决线性系统或最小二乘问题
> CV_EXPORTS_W bool solve(InputArray src1, InputArray src2, OutputArray dst, int flags = DECOMP_LU);
>
> // 计算对称矩阵的特征值和特征向量
> CV_EXPORTS_W bool eigen(InputArray src, OutputArray eigenvalues, OutputArray eigenvectors = noArray());
> ```
## 6. 常用的辅助函数
> ```
> // 多线程与opencv内核设置
> cv::setNumThreads(4);
> int threads = cv::getNumThreads();
> cv::useOptimized(true);
>
> // 调试辅助函数
> cv::namedWindow("debug")
> cv::imshow("debug",mat)
> cv::waitKey(0);
> cv::destroyAllWindows();
> cv::imwrite("debug.jpg",mat);
>
> // 随机生成图像
> // 生成一个均匀分布的随机数或随机数数组
> // dst: 随机数输出数组，必须预先分配
> // low: 生成随机数的包含下限
> // high: 生成随机数的排他性上限
> CV_EXPORTS_W void randu(InputOutputArray dst, InputArray low, InputArray high);
> // 用正态分布随机填充数组
> // dst: 随机数输出数组，必须预先分配，并且具有1~4个信道
> // mean: 生成的随机数的平均值(期望值)
> // stddev: 生成随机数的标准偏差，可以是向量
> CV_EXPORTS_W void randn(InputOutputArray dst, InputArray mean, InputArray stddev);
>
> // 高精度计时
> CV_EXPORTS_W int64 getTickCount();
> CV_EXPORTS_W double getTickFrequency();
> 
> ```



# 二、函数
## 1. 图像滤波器
|任务|推荐滤波器|原因|
|:--:|:--:|:--:|
|金属表面划痕|高斯滤波+Laplacian|去噪+边缘增强|
|胶囊缺口检测|中值滤波+Canny|去噪+边缘提取清晰|
|背景建模(光照变化)|高斯滤波/双边滤波|平滑不均匀背景|
|OCR提取|自适应高斯+Sobel|背景压制+字符突出|
|激光线提取|Gabor/Sobel(垂直)|精准提取单方向结构|
|实时视频降噪|中值滤波(高速)/时域滤波|快速去除噪点，实时处理|

### 1.1 平滑/去噪滤波器
#### 1.1.1 均值滤波 Mean
> 
#### 1.1.2 高斯滤波 GaussianBlur
> **CV_EXPORTS_W void GaussianBlur( InputArray src, OutputArray dst, Size ksize,double sigmaX, double sigmaY = 0, int borderType = BORDER_DEFAULT );**    
> 使用高斯滤波器对图像进行卷积。支持**就地滤波**    
> - src: 输入图像，可任意通道数，每个通道独立处理，深度应为CV_8U, CV_16U, CV_16S, CV_32F, CV_64F    
> - dst: 输出图像，与src大小和类型相同    
> - ksize: 高斯核大小。ksize.width和ksize.height可以不同，但必须为正奇数。或者，它们可以为零，然后根据sigma计算得出    
> - sigmaX：高斯核在X方向上的标准差    
> - sigmaY: 高斯核在Y方向上的标准差，如果simgaY为零，则将其设置为等于simgaX;如果俩个都为零，则分别根据ksize.width和ksize.height计算得出    
> - borderType: 详见BorderTypes,目前运行效果看，应该是线条的宽度
> 
> |*sigmaX*|*模糊程度*|*图像表现*|
> |:--:|:--:|:--:|
> |0|不模糊(无效果)|图像保持原样|
> |~0.3|非常轻微|几乎无变化|
> |1.0|中度模糊|一般模糊，部分边缘保留|
> |2.0|强模糊|细节严重丢失，边缘柔化|
> |5.0+|极强模糊|类似高斯光晕，结构消失|
> 
> 工业场景建议
> |*应用场景*|*推荐sigmaX*|
> |:--:|:--:|
> |表面噪声去除|0.5~1.0|
> |背景建模模糊|1.0~2.5|
> |高斯差分检测|0.5~2.0|
> |光照不均处理前预平滑|2.0+|

#### 1.1.3 中值滤波 Median
#### 1.1.4 双边滤波 Bilateral
#### 1.1.5 自定义核滤波
### 1.2 锐化滤波器
#### 1.2.1 拉普拉斯 Laplacian 
#### 1.2.2 Sobel 
#### 1.2.3 Scharr
#### 1.2.4 锐化核
### 1.3 边缘检测滤波器
#### 1.3.1 Canny边缘检测
#### 1.3.2 Sobel/Scharr
#### 1.3.3 Laplacian 
#### 1.3.4 Prewitt/Roberts
### 1.4 频域滤波
#### 1.4.1 低通滤波器
#### 1.4.2 高通滤波器
#### 1.4.3 带通/带阻滤波器
### 1.5 方向滤波/特殊滤波器
#### 1.5.1 方向增强滤波器 
#### 1.5.2 Gabor滤波器
#### 1.5.3 DoG 高斯差分
