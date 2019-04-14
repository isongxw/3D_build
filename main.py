'''
marching_cubes_lewiner函数中的level参数使用的是实际像素值，CT图像中使用的是HU值，在使用level参数时需要将HU值转换为像素值，
转换公式为 Hu=pixel_val*RescaleSlope+RescaleIntercept，其中RescaleSlope与RescaleIntercept都是DICOM文件中的tag值，
此次使用的数据中RescaleSlope=1，RescaleIntercept=-1024，因此mimics中骨骼HU值为226，转换为level值为1250.

measure.label()函数 默认情况下标记像素值0为背景像素，并标记他们为0
'''
from shutil import copyfile
import win32api
import datetime
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
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

sec = 0
data_path = ""
smooth_sel = 0
fileList = []
timer = QTimer()

def add_log(self, log):
    self.console_browser.append(datetime.datetime.now().strftime('%y-%b-%d'
                                                                 ' %H:%M:%S') + ': ' + str(log))
    #QtWidgets.QApplication.processEvents()


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
        temp = gaussian_filter(image,1)

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
    verts, faces= measure.marching_cubes_classic(p)
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
    resolution_w = win32api.GetSystemMetrics(0)
    resolution_h = win32api.GetSystemMetrics(1)
    stlreader = vtk.vtkSTLReader()
    stlreader.SetFileName(stl_name)
    cylinderMapper = vtk.vtkPolyDataMapper()  # 渲染多边形几何数据
    cylinderMapper.SetInputConnection(stlreader.GetOutputPort())  # VTK可视化管线的输入数据接口 ，对应的可视化管线输出数据的接口为GetOutputPort()；
    cylinderActor = vtk.vtkActor()
    cylinderActor.SetMapper(cylinderMapper)  # 设置生成几何图元的Mapper。即连接一个Actor到可视化管线的末端(可视化管线的末端就是Mapper)。
    renderer = vtk.vtkRenderer()  # 负责管理场景的渲染过程
    renderer.AddActor(cylinderActor)
    renderer.SetBackground(0.1, 0.1, 0.4)
    renWin = vtk.vtkRenderWindow()  # 将操作系统与VTK渲染引擎连接到一起。
    renWin.AddRenderer(renderer)
    renWin.SetSize(800, 800)
    renWin.SetPosition(int(resolution_w/2 - 400), int(resolution_h/2 - 400))
    renWin.Render()
    renWin.SetWindowName("Preview Window")
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


class WorkThread(QThread):
    # 定义一个新的线程类
    trigger = pyqtSignal()

    def __int__(self):
        super(WorkThread, self).__init__()

    def run(self):
        # 重写run函数，进程start()时调用run()函数
        global data_path
        global smooth_sel
        global fileList
        print("正在扫描数据...")
        patient = load_scan(data_path, fileList)
        print("数据扫描完成")
        imgs_to_process = np.stack([s.pixel_array for s in patient])
        print("正在进行数据预处理...")
        imgs_to_smooth = region_grow(imgs_to_process, 2)
        print("数据预处理完成")
        print("正在进行数据平滑...")
        imgs_after_smooth = smooth_data(imgs_to_smooth, smooth_sel)
        #add_log(self, "数据平滑完成，数据平滑方式为:" + str(self.smooth_sel))
        #add_log(self, "正在进行数据重采样...")
        imgs_after_resamp = resample(imgs_after_smooth, patient, [1, 1, 1])
        # imgs_after_resamp = imgs_after_smooth
        print("数据重采样完成")
        # 在三维数据的顶部与底部添加全0的二维数组，人为创建等值面进行三维重建以填补空白
        zero = np.zeros_like(imgs_after_resamp[0])
        imgs_after_resamp = imgs_after_resamp.tolist()
        zero = zero.tolist()
        imgs_after_resamp.append(zero)
        imgs_after_resamp.insert(0, zero)
        imgs_after_resamp = np.array(imgs_after_resamp)
        np.save("imgs_after_resamp.npy", imgs_after_resamp)
        print("数据处理完成")
        self.trigger.emit()

class mywindows(QtWidgets.QMainWindow, Qt_GUI.Ui_mainWindow):

    default_path = os.path.abspath('.')
    default_name = 'preview.stl'
    work = WorkThread()

    def __init__(self):
        super(mywindows, self).__init__()
        self.setupUi(self)

    def countTime(self):
        # 每经过一秒进行一次处理
        global sec
        sec += 1
        self.lcdNumber.display(sec)
        if sec < 297:
            # 每经过三秒进度条加一，按照经验来看，数据处理一般在300秒之内
            self.progressBar.setValue(int(sec / 3))

    def workEnd(self):
        # 线程完成后的工作
        global timer
        global sec
        sec = 0
        timer.stop()
        self.progressBar.setValue(100)
        add_log(self, "数据处理完成，用时:" + str(self.lcdNumber.value()) + "秒")
        QMessageBox.information(self, "数据处理", "数据处理完成", QMessageBox.Ok)
        self.progressBar.setVisible(False)
        self.pushButton.setEnabled(True)

    def get_smooth(self, sel):
        global smooth_sel
        smooth_sel = sel
    def get_path(self):
        global data_path
        global fileList
        data_path = QFileDialog.getExistingDirectory(self, "choose origin directory", "./")
        fileList = glob(data_path + '/*.dcm')  # 得到路径下所有的.dcm文件，返回类型为list
        if(len(fileList) == 0):
            QMessageBox.warning(self, "warning", "未发现符合要求的文件", QMessageBox.Ok)
        else:
            self.pushButton.setDisabled(True)
            add_log(self, "源文件路径为: " + data_path)
            QMessageBox.information(self, "文件扫描", "文件扫描完成，包含%d个.dcm文件" % (len(fileList)), QMessageBox.Ok)
            add_log(self, "文件扫描完成，包含%d个.dcm文件" % (len(fileList)))


    def data_proc(self):
        global timer
        if(data_path == ""):
            QMessageBox.warning(self, "warning", "未选择文件夹", QMessageBox.Ok)
        else:
            add_log(self, "开始进行数据处理...")
            # 开始计时，时长为1s
            timer.start(1000)
            # 启动线程
            self.work.start()
            self.progressBar.setVisible(True)
            # 捕获进程结束信号
            self.work.trigger.connect(self.workEnd)
            # 捕获一次计时结束信号
            timer.timeout.connect(self.countTime)


    def generate_stl(self):
        # 生成stl
        stl_path = QFileDialog.getExistingDirectory(self, "choose output directory", "./")
        add_log(self, "输出文件路径为: " + stl_path)
        stl_name = self.filename_edit.text()
        srcfile = self.default_path + '/' + self.default_name
        dstfile = stl_path + '/' + stl_name + '.stl'
        if(stl_name == ""):
            QMessageBox.warning(self, "warning", "未输入.stl文件名", QMessageBox.Ok)
        else:
            copyfile(srcfile, dstfile)
            add_log(self, "生成stl文件完成")
            QMessageBox.information(self, "生成stl", "生成stl文件完成", QMessageBox.Ok)

    def preview_3D(self):
        imgs_used = np.load("imgs_after_resamp.npy")
        add_log(self, "正在生成预览...")
        v, f = make_mesh(imgs_used)
        three_d_print(v, f, self.default_path, self.default_name)
        filename = self.default_path + '/' + self.default_name
        preview(filename)


app = QtWidgets.QApplication(sys.argv)
window = mywindows()
# ui = Qt_GUI.Ui_MainWindow()
# ui.setupUi(window)
window.show()
sys.exit(app.exec_())
