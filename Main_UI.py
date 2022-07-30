import sys
import warnings
import cv2
import numpy as np
import struct
import matplotlib.pyplot as plt

from PyQt5 import QtWidgets, QtGui

from ColorMenu import ColorMenu
from geometryMenu import geometryMenu
from mophologyMenu import mophologyMenu
from edgeMenu import EdgeMenu
from filterMenu import FilterMenu
from faceDetect import FaceDetect
from ui_file_1 import  Ui_mainWindow
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QInputDialog, QLineEdit
from qt_material import apply_stylesheet


class WXTWindow(QtWidgets.QMainWindow, Ui_mainWindow):
    def __init__(self):
        self.default_save_path = 'img'
        super(WXTWindow, self).__init__()
        self.setupUi(self)
        #QMessageBox.about(self.window(), "使用提示", "欢迎使用~ 在对图像进行操作前，请先选择要处理的图片~")
        self.pic_path = ""  # 初始化空值
        self.menubar.setEnabled(True)
        self.color_menu.setEnabled(False)
        self.geo_menu.setEnabled(False)
        self.histogram_menu.setEnabled(False)
        self.mophology_menu.setEnabled(False)
        self.advancecd_menu.setEnabled(False)
        self.denoising_menu.setEnabled(False)
        self.faceMenu.setEnabled(False)
        self.Save_btn.setEnabled(False)

    # 打开图片
    def OpenImg(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "OriginPictures", "*.jpg;;*.bmp;;*.png")
        jpg = QtGui.QPixmap(imgName).scaled(self.ori_pic.width(), self.ori_pic.height(), 1 ,1)
        self.ori_pic.setPixmap(jpg)

        self.origin_pic_path.setText(imgName)
        self.pic_path = imgName
        self.fixed_pic.setText("")
        self.fixed_pic.clear()

        if self.pic_path == "":
            self.color_menu.setEnabled(False)
            self.geo_menu.setEnabled(False)
            self.histogram_menu.setEnabled(False)
            self.mophology_menu.setEnabled(False)
            self.advancecd_menu.setEnabled(False)
            self.denoising_menu.setEnabled(False)
            self.faceMenu.setEnabled(False)
            self.Save_btn.setEnabled(False)
            return

        #各部分初始化
        self.Initialize_color()
        self.Initialize_geo()
        self.InitializeMopho()
        self.InitializeEdge()
        self.InitializeFilter()

        self.color_menu.setEnabled(True)
        self.geo_menu.setEnabled(True)
        self.histogram_menu.setEnabled(True)
        self.mophology_menu.setEnabled(True)
        self.advancecd_menu.setEnabled(True)
        self.denoising_menu.setEnabled(True)
        self.faceMenu.setEnabled(True)
        self.Save_btn.setEnabled(True)

    def SaveImg(self):
        if self.save_pic_path.text() == "默认保存在img文件夹":
            QMessageBox.information(self, "提示", "尚未进行图片操作，不能保存")
            return

        if self.pic_path != "" :
            imgURL = QFileDialog.getExistingDirectory(self, "保存图片", "SavedPictures")
            savepic = cv2.imread(self.save_pic_path.text(), 1)
            cv2.imwrite(imgURL + '/SavedPic.jpg', savepic)


    # 显示保存的图片
    def Show_Fixed_Pic(self, path):
        jpg = QtGui.QPixmap(path).scaled(self.fixed_pic.width(), self.fixed_pic.height(), 1, 1)
        self.fixed_pic.setPixmap(jpg)
        self.save_pic_path.setText(path)

    # 显示位图信息头
    def Show_Pic_Info(self):
        pass



    #以下为色彩菜单函数
    def Initialize_color(self):
        self.color = ColorMenu(self.pic_path)

    def GreyPic(self):
        greypic = cv2.imread(self.pic_path, 0)
        cv2.imwrite('img/greyimg.jpg', greypic)
        self.Show_Fixed_Pic('img/greyimg.jpg')

    def ExtractBchannel(self):
        self.color.Bchannel()
        self.Show_Fixed_Pic('img/colorImgs/Cchannel.jpg')

    def ExtractGchannel(self):
        self.color.Gchannel()
        self.Show_Fixed_Pic('img/colorImgs/Cchannel.jpg')

    def ExtractRchannel(self):
        self.color.Rchannel()
        self.Show_Fixed_Pic('img/colorImgs/Cchannel.jpg')

    def ExtractHchannel(self):
        self.color.Hchannel()
        self.Show_Fixed_Pic('img/colorImgs/Cchannel.jpg')

    def ExtractSchannel(self):
        self.color.Schannel()
        self.Show_Fixed_Pic('img/colorImgs/Cchannel.jpg')

    def ExtractVchannel(self):
        self.color.Vchannel()
        self.Show_Fixed_Pic('img/colorImgs/Cchannel.jpg')



    #以下为移动菜单函数

    def Initialize_geo(self):
        self.geoOperation = geometryMenu(self.pic_path)

    def MovePic(self):
        numx, ok1 = QInputDialog.getInt(self, 'X轴移动值', '请输入向右移动的值', 0, -10000, 10000, 10)
        numy, ok2 = QInputDialog.getInt(self, 'Y轴移动值', '请输入向下移动的值', 0, -10000, 10000, 10)

        if ok1 and ok2:
            #if numx != 0 or numy != 0:
            self.geoOperation.ImageMove(numx, numy)
            self.Show_Fixed_Pic('img/geometryImgs/moved_pic.jpg')

    def ImageResizeTimes(self):
        numx, ok1 = QInputDialog.getDouble(self, '横向放缩值', '请输入横向放缩值', 1, 0.01, 5, 1)
        numy, ok2 = QInputDialog.getDouble(self, '纵向放缩值', '请输入纵向放缩值', 1, 0.01, 5, 1)

        if ok1 and ok2:
            if numx != 1 or numy != 1:
                self.geoOperation.ImageResizeTimes(numx, numy)
                self.Show_Fixed_Pic('img/geometryImgs/moved_pic.jpg')

    def ImageResizePixel(self):
        numx, ok1 = QInputDialog.getInt(self, '横向像素值', '请输入横向像素值', 8, 8, 20000, 1)
        numy, ok2 = QInputDialog.getInt(self, '纵向像素值', '请输入纵向像素值', 8, 8, 20000, 1)

        if ok1 and ok2:
            self.geoOperation.ImageResizePixel(numx, numy)
            self.Show_Fixed_Pic('img/geometryImgs/moved_pic.jpg')

    def HorizontalFlip(self):
        self.geoOperation.HorizentalFlip()
        self.Show_Fixed_Pic('img/geometryImgs/moved_pic.jpg')

    def VerticalFlip(self):
        self.geoOperation.VerticalFlip()
        self.Show_Fixed_Pic('img/geometryImgs/moved_pic.jpg')

    def CrossFlip(self):
        self.geoOperation.CrossFlip()
        self.Show_Fixed_Pic('img/geometryImgs/moved_pic.jpg')

    def FreeRotate(self):
        angle, ok1 = QInputDialog.getInt(self, '角度', '请输入顺时针旋转角度', 0, -360, 360, 1)
        size, ok2 = QInputDialog.getDouble(self, '放缩值', '请输入放缩值,默认为1无需更改', 1, 0.01, 5, 1)

        if ok1 and ok2:
            self.geoOperation.FreeRotate(angle, size)
            self.Show_Fixed_Pic('img/geometryImgs/moved_pic.jpg')

    def FixedRotate(self):
        items = ('90', '180', '270')
        angle, ok = QInputDialog.getItem(self, '角度', '请输入顺时针旋转角度', items, 0, False)

        if ok:
            self.geoOperation.FixedRotate(angle)
            self.Show_Fixed_Pic('img/geometryImgs/moved_pic.jpg')



    #以下为直方图函数
    def DrawGreyHistogram(self):
        greypic = cv2.imread(self.pic_path, 0)
        hist = cv2.calcHist([greypic], [0], None, [256], [0, 256])
        plt.plot(hist)
        plt.xlim([0, 255])
        plt.savefig('img/histogram/histogram.jpg')
        plt.close()
        self.Show_Fixed_Pic('img/histogram/histogram.jpg')

    def DrawRGBHistogram(self):
        pic = cv2.imread(self.pic_path, 1)
        color = ('r', 'g', 'b')
        for i, col in enumerate(color):
            hist = cv2.calcHist([pic], [i], None, [256], [0, 256])
            plt.plot(hist)
            plt.xlim([0, 256])

        plt.savefig('img/histogram/histogram.jpg')
        plt.close()
        self.Show_Fixed_Pic('img/histogram/histogram.jpg')



    #以下为形态学运算函数
    def InitializeMopho(self):
        self.mophology = mophologyMenu(self.pic_path)

    def Erode(self):
        size, ok = QInputDialog.getInt(self, '腐蚀结构元大小', '请输入，默认为5', 5, 1, 1000, 1)

        if ok:
            self.mophology.Erode(size)
            self.Show_Fixed_Pic('img/mophologyImgs/mophoPic.jpg')

    def Dilate(self):
        size, ok = QInputDialog.getInt(self, '膨胀结构元大小', '请输入，默认为5', 5, 1, 1000, 1)

        if ok:
            self.mophology.Dilate(size)
            self.Show_Fixed_Pic('img/mophologyImgs/mophoPic.jpg')

    def OpenOperate(self):
        size, ok = QInputDialog.getInt(self, '开运算结构元大小', '请输入，默认为5', 5, 1, 1000, 1)

        if ok:
            self.mophology.Open(size)
            self.Show_Fixed_Pic('img/mophologyImgs/mophoPic.jpg')
    def CloseOperate(self):
        size, ok = QInputDialog.getInt(self, '闭运算结构元大小', '请输入，默认为5', 5, 1, 1000, 1)

        if ok:
            self.mophology.Close(size)
            self.Show_Fixed_Pic('img/mophologyImgs/mophoPic.jpg')



    #以下为图像增强以及边缘检测、锐化函数
    def InitializeEdge(self):
        self.edgeOpe = EdgeMenu(self.pic_path)

    def PicStrength(self):
        self.edgeOpe.PicStrength()
        self.Show_Fixed_Pic('img/EdgeImgs/EdgePic.jpg')

    def RobertsOperation(self):
        self.edgeOpe.EdgeRoberts()
        self.Show_Fixed_Pic('img/EdgeImgs/EdgePic.jpg')

    def SobelOperation(self):
        self.edgeOpe.EdgeSobel()
        self.Show_Fixed_Pic('img/EdgeImgs/EdgePic.jpg')

    def LaplacianOperation(self):
        items = ('1', '3', '5', '7')
        size, ok = QInputDialog.getItem(self, 'Laplacian算子大小', '请输入，默认为1', items, 0, False)

        if ok:
            self.edgeOpe.EdgeLaplacian(size)
            self.Show_Fixed_Pic('img/EdgeImgs/EdgePic.jpg')

    def LoGOperation(self):
        QMessageBox.information(self, "提示", "LoG算子运算较慢请耐心等待")

        self.edgeOpe.EdgeLoG()
        self.Show_Fixed_Pic('img/EdgeImgs/EdgePic.jpg')

    def CannyOperation(self):
        self.edgeOpe.EdgeCanny()
        self.Show_Fixed_Pic('img/EdgeImgs/EdgePic.jpg')

    def HoughStraight(self):
        colors = ['red', 'green', 'blue', 'black', 'white']
        color, ok1 = QInputDialog.getItem(self, '线条颜色', '请输入，默认为红', colors, 0, False)
        thres, ok2 = QInputDialog.getInt(self, '阈值', '请输入阈值，阈值越大检测出的线段越长越少，默认为100', 100, 1, 1000, 1)

        if ok1 and ok2:
            self.edgeOpe.HoughDetect(thres, color)
            self.Show_Fixed_Pic('img/EdgeImgs/EdgePic.jpg')

    def HoughCurve(self):
        colors = ['red', 'green', 'blue', 'black', 'white']
        color, ok1 = QInputDialog.getItem(self, '线条颜色', '请输入，默认为红', colors, 0, False)
        thres, ok2 = QInputDialog.getInt(self, '阈值', '请输入阈值，阈值越大检测出的线段越长越少，默认为100', 100, 1, 1000, 1)
        minline, ok3 = QInputDialog.getInt(self, '线段最小长度', '请输入线段最小长度，默认为0', 0, 0, 1000, 1)
        maxgap, ok4 = QInputDialog.getInt(self, '线段间最大允许间隔', '请输入间隔默认为0', 0, 0, 1000, 1)

        if ok1 and ok2 and ok3 and ok4:
            self.edgeOpe.HoughDetectP(thres, minline, maxgap, color)
            self.Show_Fixed_Pic('img/EdgeImgs/EdgePic.jpg')






    #以下为去噪相关部分槽函数
    def InitializeFilter(self):
        self.filterOpe = FilterMenu(self.pic_path)

    def GaussNoise(self):
        avg, ok1 = QInputDialog.getDouble(self, '均值', '请输入均值', 0, 0, 100)
        var, ok2 = QInputDialog.getDouble(self, '方差', '请输入方差', 0.5, 0, 100)

        if ok1 and ok2:
            self.filterOpe.GaussNoise(avg,var)
            self.Show_Fixed_Pic('img/filterImgs/FilterPic.jpg')

    def PepperSaltyNoise(self):
        thres, ok1 = QInputDialog.getDouble(self, '噪声阈值', '请输入阈值', 0.2, 0.01, 0.99)

        if ok1:
            self.filterOpe.PepperSaltyNoise(thres)
            self.Show_Fixed_Pic('img/filterImgs/FilterPic.jpg')

    def GaussFilter(self):
        sizes, ok = QInputDialog.getInt(self, '高斯核大小', '请输入,必须为奇数,默认为1', 1, 1, 15)

        if ok:
            if sizes % 2 == 1:
                self.filterOpe.gaussFilter(sizes)
                self.Show_Fixed_Pic('img/filterImgs/FilterPic.jpg')
            else:
                QMessageBox.information(self, "提示", "输入的核大小必须为奇数")

    def AvgFilter(self):
        self.filterOpe.AverageFilter()
        self.Show_Fixed_Pic('img/filterImgs/FilterPic.jpg')

    def MidFilter(self):
        self.filterOpe.middleFilter()
        self.Show_Fixed_Pic('img/filterImgs/FilterPic.jpg')

    def minFilter(self):
        self.filterOpe.LeastFilter()
        self.Show_Fixed_Pic('img/filterImgs/FilterPic.jpg')

    def maxFilter(self):
        self.filterOpe.LargestFilter()
        self.Show_Fixed_Pic('img/filterImgs/FilterPic.jpg')

    def PassFilter(self):
        items = ('0', '255')

        min, ok1 = QInputDialog.getInt(self, '带通范围', '请输入最低范围,至少0', 20, 0, 255)
        max, ok2 = QInputDialog.getInt(self, '带通范围', '请输入最高范围,至多255', 220, 0, 255)
        num, ok3 = QInputDialog.getItem(self, '非滤指定值', '请选择', items, 0, False)

        if ok1 and ok2 and ok3:
            self.filterOpe.PassFilter(min, max, num)
            self.Show_Fixed_Pic('img/filterImgs/FilterPic.jpg')

    def BlockFilter(self):
        items = ('0', '255')

        min, ok1 = QInputDialog.getInt(self, '带阻范围', '请输入最低范围,至少0', 20, 0, 255)
        max, ok2 = QInputDialog.getInt(self, '带阻范围', '请输入最高范围,至多255', 220, 0, 255)
        num, ok3 = QInputDialog.getItem(self, '非滤指定值', '请选择', items, 0, False)

        if ok1 and ok2 and ok3:
            self.filterOpe.BlockFilter(min, max, num)
            self.Show_Fixed_Pic('img/filterImgs/FilterPic.jpg')

    #以下是人脸识别
    def faceDetect(self):
        self.facedtct = FaceDetect(self.pic_path)
        self.facedtct.detectFace()
        self.Show_Fixed_Pic('img/faceImg/facePic.jpg')

    def videoDetect(self):
        self.facedtct = FaceDetect(self.pic_path)
        try:
            self.facedtct.videoDetect()
        except Exception as e:
            QMessageBox.information(self, "提示", "退出了人脸识别")






if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    app = QtWidgets.QApplication(sys.argv)
    myWin = WXTWindow()
    apply_stylesheet(app, theme='light_cyan.xml')
    myWin.show()
    sys.exit(app.exec_())