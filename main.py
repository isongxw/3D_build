'''
marching_cubes_lewiner函数中的level参数使用的是实际像素值，CT图像中使用的是HU值，在使用level参数时需要将HU值转换为像素值，
转换公式为 Hu=pixel_val*RescaleSlope+RescaleIntercept，其中RescaleSlope与RescaleIntercept都是DICOM文件中的tag值，
此次使用的数据中RescaleSlope=1，RescaleIntercept=-1024，因此mimics中骨骼HU值为226，转换为level值为1250.

measure.label()函数 默认情况下标记像素值0为背景像素，并标记他们为0
'''

import datetime
from skimage import measure
from stl import mesh
from PyQt5 import QtWidgets, QtGui
import sys
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QApplication
import Qt_GUI
import numpy as np
import pydicom
import os
import scipy.ndimage
from glob import glob
from scipy.ndimage import gaussian_filter, median_filter
import vtk


def add_log(self, log):
    self.console_browser.append(datetime.datetime.now().strftime('%y-%b-%d'
                                                                 ' %H:%M:%S') + ': ' + str(log))
    QtWidgets.QApplication.processEvents()


def load_scan(path, fileList):
    slices = [pydicom.dcmread(s) for s in fileList]
    slices.sort(key=lambda x: int(x.InstanceNumber), reverse=True)  # 将slices按照.dcm文件的InstanceNumber属性进行排序
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    for s in slices:
        s.SliceThickness = slice_thickness

    return slices


def smooth_data(image, sel):
    if (sel == 0):
        return image
    if (sel == 1):
        temp = np.zeros(shape=image.shape)
        for thickness in range(image.shape[0]):
            temp[thickness] = gaussian_filter(image[thickness], sigma=1)
    else:
        temp = gaussian_filter(image, 1)

    return temp


def region_grow(img, cnct = 2):
    img = measure.label(img, connectivity=cnct)
    max_num = 0
    max_label = -1
    max_region = np.max(img)
    for i in range(1, max_region + 1):
        if np.sum(img==i) > max_num:
            max_num = np.sum(img==i)
            max_label = i
            QApplication.processEvents()
    result = img==max_label
    result = result*255
    return result


def resample(image, scan, new_spacing):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + list(scan[0].PixelSpacing)))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    return image


def make_mesh(image, step_size=1):
    p = image.transpose(2, 1, 0)
    verts, faces , d , e= measure.marching_cubes_lewiner(p, level = 100, allow_degenerate=True)
    return verts, faces


def three_d_print(verts, faces, output_path, name):
    solid = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces):
        for j in range(3):
            solid.vectors[i][j] = verts[f[j], :]

    solid.save(output_path + '/' + name)


def preview(stl_name):
    if stl_name == "":
        return
    stlreader = vtk.vtkSTLReader()
    stlreader.SetFileName(stl_name)
    cylinderMapper = vtk.vtkPolyDataMapper()  # 渲染多边形几何数据
    cylinderMapper.SetInputConnection(stlreader.GetOutputPort())  # VTK可视化管线的输入数据接口 ，对应的可视化管线输出数据的接口为GetOutputPort()；
    cylinderActor = vtk.vtkActor()
    cylinderActor.SetMapper(cylinderMapper)  # 设置生成几何图元的Mapper。即连接一个Actor到可视化管线的末端(可视化管线的末端就是Mapper)。
    renderer = vtk.vtkRenderer()  # 负责管理场景的渲染过程
    renderer.AddActor(cylinderActor)
    renderer.SetBackground(0.1, 0.2, 0.4)
    renWin = vtk.vtkRenderWindow()  # 将操作系统与VTK渲染引擎连接到一起。
    renWin.AddRenderer(renderer)
    renWin.SetSize(800, 800)
    iren = vtk.vtkRenderWindowInteractor()  # 提供平台独立的响应鼠标、键盘和时钟事件的交互机制
    iren.SetRenderWindow(renWin)
    # 交互器样式的一种，该样式下，用户是通过控制相机对物体作旋转、放大、缩小等操作
    style = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(style)
    iren.Initialize()
    iren.Start()
    # Clean up
    # del cylinder
    del stlreader
    del cylinderMapper
    del cylinderActor
    del renderer
    del renWin
    del iren




class mywindows(QtWidgets.QMainWindow, Qt_GUI.Ui_mainWindow):

    data_path = ""
    smooth_sel = 0
    stl_name = ""
    stl_path = ""
    fileList = []
    def __init__(self):
        super(mywindows, self).__init__()
        self.setupUi(self)

    def get_path(self):
        self.data_path = QFileDialog.getExistingDirectory(self, "choose origin directory", "./")
        self.fileList = glob(self.data_path + '/*.dcm')  # 得到路径下所有的.dcm文件，返回类型为list
        add_log(self, "源文件路径为: " + self.data_path)
        QMessageBox.information(self, "文件扫描", "文件扫描完成，包含%d个.dcm文件" % (len(self.fileList)), QMessageBox.Ok)
        add_log(self, "文件扫描完成，包含%d个.dcm文件" % (len(self.fileList)))

    def get_smooth(self, sel):
        self.smooth_sel = sel

    def data_proc(self):
        if(self.data_path == ""):
            QMessageBox.warning(self, "warning", "未选择文件夹", QMessageBox.Ok)
        else:
            add_log(self, "正在扫描数据...")
            patient = load_scan(self.data_path, self.fileList)
            add_log(self, "数据扫描完成")
            imgs_to_process = np.stack([s.pixel_array for s in patient])
            add_log(self, "正在进行数据预处理...")
            imgs_to_smooth= region_grow(imgs_to_process, 2)
            add_log(self, "数据预处理完成")
            add_log(self, "正在进行数据平滑...")
            imgs_after_smooth = smooth_data(imgs_to_smooth, self.smooth_sel)
            add_log(self, "数据平滑完成，数据平滑方式为:"+ str(self.smooth_sel))
            add_log(self, "正在进行数据重采样...")
            imgs_after_resamp = resample(imgs_after_smooth, patient, [1, 1, 1])
            add_log(self, "数据重采样完成")
            np.save("imgs_after_resamp.npy", imgs_after_resamp)
            add_log(self, "数据处理完成")
            QMessageBox.information(self, "数据处理", "数据处理完成", QMessageBox.Ok)

    def generate_stl(self):
        self.stl_path = QFileDialog.getExistingDirectory(self, "choose output directory", "./")
        add_log(self, "输出文件路径为: " + self.stl_path)
        self.stl_name = self.filename_edit.text()
        if(self.stl_name == ""):
            QMessageBox.warning(self, "warning", "未输入.stl文件名", QMessageBox.Ok)
        else:
            imgs_used = np.load("imgs_after_resamp.npy")
            add_log(self, "开始制作网格...")
            v, f = make_mesh(imgs_used)
            add_log(self, "制作网格完成")
            filename = self.stl_name + '.stl'
            add_log(self, "开始写入stl文件...")
            three_d_print(v, f, self.stl_path, filename)
            add_log(self, "写入stl文件完成")
            QMessageBox.information(self, "生成stl", "生成stl文件完成", QMessageBox.Ok)

    def preview_3D(self):
        if self.stl_name == "":
            QMessageBox.warning(self, "warning", "未生成.stl文件", QMessageBox.Ok)
        else:
            filename = self.stl_path + '/'+ self.stl_name + '.stl'
            preview(filename)


app = QtWidgets.QApplication(sys.argv)
window = mywindows()
# ui = Qt_GUI.Ui_MainWindow()
# ui.setupUi(window)
window.show()
sys.exit(app.exec_())
