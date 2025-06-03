# 目录
- [一、OpenCV的主要模块及核心简介](#一OpenCV的主要模块及核心简介)
- [二、Core模块(Core)](#二Core模块-Core-)
  - [1. 图像滤波器](#1-图像滤波器)
    - [1.1 平滑/去噪滤波器](#11-平滑去噪滤波器)
    - [1.2 锐化滤波器](#12-锐化滤波器)
    - [1.3 边缘检测滤波器](#13-边缘检测滤波器)
    - [1.4 频域滤波](#14-频域滤波)
    - [1.5 方向滤波/特殊滤波器](#15-方向滤波特殊滤波器)
# 一、OpenCV的主要模块及核心简介
> - [ ] Core模块(Core)
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
## 1 cv::Mat 图像与多维矩阵的核心
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
