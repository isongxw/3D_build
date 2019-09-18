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
# from anisotropic_diffusion import anisodiff2D
from medpy.filter.smoothing import anisotropic_diffusion
sec = 0
data_path = ""
smooth_sel = 0
fileList = []
timer = QTimer()
default_path = os.path.abspath('.')
default_name = 'preview.stl'
func_type = ""      # 调用PDThread类的函数
dstfile = ""        # 生成的stl文件名
ImageOrientation = []   # 图像方向,格式为六个数组成的list，前三个表示第一行相对于患者在x,y,z三个方向上的余弦值，
                        # 后三个表示第一列相对于患者在x,y,z三个方向上的余弦值，由于第一行垂直于y轴，且为二维数据，
                        # 所以在前三个之后x方向上有值，同理后三个只有y方向上有值。

def add_log(self, log):
    self.console_browser.append(datetime.datetime.now().strftime('%y-%b-%d'
                                                                 ' %H:%M:%S') + ': ' + str(log))
def load_scan(path, fileList):
    global ImageOrientation
    slices = [pydicom.dcmread(s) for s in fileList]
    slices.sort(key=lambda x: int(x.InstanceNumber), reverse=True)  # 将slices按照.dcm文件的InstanceNumber属性进行排序
    ImageOrientation = slices[0].ImageOrientationPatient
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
    if (sel == 2):
        temp = gaussian_filter(image,1)
    else:
        temp = anisotropic_diffusion(image, niter = 5, kappa=300)

    return temp


def region_grow(img, cnct = 2):
    print(img.max())
    img = 3000*(img>=1250)
    result = np.zeros_like(img)
    print('label')
    img1 = measure.label(img, connectivity=cnct)
    print('region_grow')
    res = measure.regionprops(img1)
    print('rg fin')
    max_label = 0
    max_area = 0
    for re in res:
        if re.area > max_area:
            max_area = re.area
            max_label = re.label
    print(max_label, max_area)
    result = img1==max_label
    result = result*img
    return result




def resample(image, scan, new_spacing):
    # Determine current pixel spacing
    spacing = map(float, (list(scan[0].PixelSpacing)) + [scan[0].SliceThickness])

    spacing = np.array(list(spacing))
    print(spacing)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)
    return image


def make_mesh(image, step_size=1):
    global ImageOrientation
    p = image
    if ImageOrientation[0] > 0 and ImageOrientation[4] > 0:
        p = image.transpose(1, 0, 2)
    # elif ImageOrientation[0] < 0 and ImageOrientation[4] < 0:
    #     p = image

    verts, faces, n, v= measure.marching_cubes_lewiner(p, 1250)
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


class DPThread(QThread):
    # 定义一个新的线程类用于数据处理
    def __int__(self):
        super(DPThread, self).__init__()

    def run(self):
        # 重写run函数，进程start()时调用run()函数
        global data_path
        global smooth_sel
        global fileList
        print("正在扫描数据...")
        patient = load_scan(data_path, fileList)
        print("数据扫描完成")
        imgs_to_process = np.stack([s.pixel_array for s in patient], axis=2)
        print("正在进行数据预处理...")
        imgs_to_smooth = region_grow(imgs_to_process, 2)
        print("数据预处理完成")
        print("正在进行数据平滑...")
        imgs_after_smooth = smooth_data(imgs_to_smooth, smooth_sel)
        print("数据平滑完成")
        print("正在进行数据重采样...")
        imgs_after_resamp = resample(imgs_after_smooth, patient, [1, 1, 1])
        print("数据重采样完成")
        # 在三维数据的顶部与底部添加全0的二维数组，人为创建等值面进行三维重建以填补空白
        zero = np.zeros((imgs_after_resamp.shape[0], imgs_after_resamp.shape[1]))
        imgs_after_resamp = np.insert(imgs_after_resamp, 0, zero, axis=2)
        imgs_after_resamp = np.insert(imgs_after_resamp, imgs_after_resamp.shape[2], zero, axis=2)
        print(imgs_after_resamp.shape)
        # imgs_after_resamp = imgs_after_resamp.tolist()
        # zero = zero.tolist()
        # imgs_after_resamp.append(zero)
        # imgs_after_resamp.insert(0, zero)
        # imgs_after_resamp = np.array(imgs_after_resamp)
        np.save("imgs_after_resamp.npy", imgs_after_resamp)


class PDThread(QThread):
    # 定义一个新的线程类用于3D预览
    def __int__(self):
        global stl_finished
        stl_finished = False
        super(PDThread, self).__init__()

    def run(self):
        # 重写run函数，进程start()时调用run()函数
        global default_name
        global default_path
        imgs_used = np.load("imgs_after_resamp.npy")
        v, f = make_mesh(imgs_used)
        three_d_print(v, f, default_path, default_name)
        


class mywindows(QtWidgets.QMainWindow, Qt_GUI.Ui_mainWindow):

    dp_work = DPThread()
    pd_work = PDThread()

    def __init__(self):
        super(mywindows, self).__init__()
        self.setupUi(self)# 捕获进程结束信号
        self.dp_work.finished.connect(self.dp_end)
        self.pd_work.finished.connect(self.pd_end)
        # 捕获一次计时结束信号
        timer.timeout.connect(self.countTime)

    def countTime(self):
        # 每经过一秒进行一次处理
        global sec
        sec += 1
        self.lcdNumber.display(sec)
        if sec < 34:
            # 每经过三秒进度条加一，按照经验来看，数据处理一般在300秒之内
            self.progressBar.setValue(int(sec * 3))

    def dp_end(self):
        # 线程完成后的工作
        global timer
        global sec
        sec = 0
        timer.stop()
        self.progressBar.setValue(100)
        add_log(self, "数据处理完成，用时: " + str(self.lcdNumber.value()) + "秒")
        QMessageBox.information(self, "数据处理", "数据处理完成", QMessageBox.Ok)
        self.dp_work.quit()
        self.dp_work.wait()
        self.lcdNumber.display(0)
        self.progressBar.setValue(0)
        self.progressBar.setVisible(False)
        self.lcdNumber.setVisible(False)
        self.pushButton.setEnabled(True)
        self.generate_button.setEnabled(True)

    def pd_end(self):
        global timer
        global sec
        global func_type
        global dstfile
        global default_name
        global default_path

        sec = 0
        timer.stop()
        self.progressBar.setValue(100)
        self.pd_work.quit()
        self.pd_work.wait()
        add_log(self, "生成STL缓存文件完成，用时: " + str(self.lcdNumber.value()) + "秒")
        if(func_type == "preview"):
            filename = default_path + '/' + default_name
            preview(filename)
        elif(func_type == "generate"):
            srcfile = default_path + '/' + default_name
            copyfile(srcfile, dstfile)
            os.remove(srcfile)
            add_log(self, "生成stl文件完成")
            QMessageBox.information(self, "生成stl", "生成stl文件完成", QMessageBox.Ok)
        self.lcdNumber.display(0)
        self.progressBar.setValue(0)
        self.progressBar.setVisible(False)
        self.lcdNumber.setVisible(False)

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
        global default_name
        global default_path
        srcfile = default_path + '/' + default_name
        if os.path.isfile(srcfile):
            # 删除上次操作时生成的stl文件
            os.remove(srcfile)
        if(data_path == ""):
            QMessageBox.warning(self, "warning", "未选择文件夹", QMessageBox.Ok)
        else:
            add_log(self, "开始进行数据处理...")
            # 开始计时，时长为1s
            timer.start(1000)
            # 启动线程
            self.dp_work.start()
            self.progressBar.setVisible(True)
            self.lcdNumber.setVisible(True)

    def generate_stl(self):
        # 生成stl
        global default_path
        global default_name
        global dstfile
        global func_type
        func_type = "generate"
        stl_path = QFileDialog.getExistingDirectory(self, "choose output directory", "./")
        stl_name = self.filename_edit.text()
        if(stl_name == ""):
            QMessageBox.warning(self, "warning", "未输入.stl文件名", QMessageBox.Ok)
            return
        add_log(self, "输出文件路径为: " + stl_path)
        srcfile = default_path + '/' + default_name
        dstfile = stl_path + '/' + stl_name + '.stl'
        if os.path.isfile(srcfile):
            # 预览时生成的.stl文件存在，直接将预览时生成的文件copy，copy完成后删除预览stl文件
            copyfile(srcfile, dstfile)
            os.remove(srcfile)
        else:
            # 生成.stl文件
            global timer
            add_log(self, "正在生成缓存文件...")
            # 开始计时
            timer.start(1000)
            # 启动线程
            self.pd_work.start()
            self.progressBar.setVisible(True)
            self.lcdNumber.setVisible(True)

    def preview_3D(self):
        global timer
        global default_name
        global default_path
        global func_type
        func_type = "preview"
        add_log(self, "正在生成缓存文件...")
        # 开始计时
        timer.start(1000)
        # 启动线程
        self.pd_work.start()
        self.progressBar.setVisible(True)
        self.lcdNumber.setVisible(True)

app = QtWidgets.QApplication(sys.argv)
window = mywindows()
# ui = Qt_GUI.Ui_MainWindow()
# ui.setupUi(window)
window.show()
sys.exit(app.exec_())
