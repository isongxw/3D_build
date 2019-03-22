# 3D_build
二维DICOM图像的三维重建与可视化
## 源文件概述
main.py:功能实现
QT_GUI:qt窗体代码
## 功能概述
通过打开文件资源管理器来选择文件夹，并筛选出其中的DICOM文件，利用marching cubes算法来将二维的DICOM重建为三维DICOM图像。
marching cubes算法使用skimage.measure.marching_cubes_lewiner()进行实现，其中参数level可进行调节，本项目中采用默认
level值进行三维重建，可以通过改变level值来分割出不同的部位。
三维重建后进行可视化操作

