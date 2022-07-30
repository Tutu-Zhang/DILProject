import math
import cv2
import numpy as np
import struct
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMessageBox

class EdgeMenu:
    def __init__(self, pic_path):
        self.pic_path = pic_path
        self.pic = cv2.imread(self.pic_path, 0).astype(np.uint8) #灰度模式读取
        self.cpic = cv2.imread(self.pic_path, 1).astype(np.uint8)

    def ShowPic(self):
        cv2.imshow('当前图片', self.pic)
        cv2.waitKey(0)  # 0一直显示，直到有键盘输入。也可以是其他数字.

    def PicStrength(self):
        self.pic = self.pic.astype('float')
        row, column = self.pic.shape
        gradient = np.zeros((row, column))

        for x in range(row - 1):
            for y in range(column - 1):
                gx = abs(self.pic[x + 1, y] - self.pic[x, y])
                gy = abs(self.pic[x, y + 1] - self.pic[x, y])
                gradient[x, y] = gx + gy

        # 3. 对图像进行增强，增强后的图像变量名为sharp
        sharp = self.pic + gradient
        sharp = np.where(sharp > 255, 255, sharp)
        sharp = np.where(sharp < 0, 0, sharp)
        gradient = gradient.astype('uint8')
        sharp = sharp.astype('uint8')

        cv2.imwrite('img/EdgeImgs/EdgePic.jpg', sharp)
        cv2.waitKey(0)

    def EdgeRoberts(self):
        kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
        kernely = np.array([[0, -1], [1, 0]], dtype=int)
        x = cv2.filter2D(self.pic, cv2.CV_16S, kernelx)
        y = cv2.filter2D(self.pic, cv2.CV_16S, kernely)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

        cv2.imwrite('img/EdgeImgs/EdgePic.jpg', Roberts)
        cv2.waitKey(0)

    def EdgeSobel(self):
        x = cv2.Sobel(self.pic, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(self.pic, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

        cv2.imwrite('img/EdgeImgs/EdgePic.jpg', Sobel)
        cv2.waitKey(0)

    def EdgeLaplacian(self, k_size):
        grayImage = cv2.GaussianBlur(self.pic, (5, 5), 0, 0)
        # 3. 拉普拉斯算法
        if k_size == '1':
            dst = cv2.Laplacian(grayImage, cv2.CV_16S, ksize=1)
        elif k_size == '3':
            dst = cv2.Laplacian(grayImage, cv2.CV_16S, ksize=3)
        elif k_size == '5':
            dst = cv2.Laplacian(grayImage, cv2.CV_16S, ksize=5)
        elif k_size == '7':
            dst = cv2.Laplacian(grayImage, cv2.CV_16S, ksize=7)
        else:
            return
        # 4. 数据格式转换
        Laplacian = cv2.convertScaleAbs(dst)
        cv2.imwrite('img/EdgeImgs/EdgePic.jpg', Laplacian)
        cv2.waitKey(0)

    def EdgeLoG(self):
        img = cv2.imread(self.pic_path)
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 2. 边缘扩充处理图像
        image = cv2.copyMakeBorder(grayImage, 2, 2, 2, 2, borderType=cv2.BORDER_REPLICATE)
        image = cv2.GaussianBlur(image, (3, 3), 0, 0)
        # 3. 使用Numpy定义LoG算子
        m1 = np.array(
            [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])

        rows = image.shape[0]
        cols = image.shape[1]
        image1 = np.zeros(image.shape)

        # 4. 卷积运算
        for k in range(0, 2):
            for i in range(2, rows - 2):
                for j in range(2, cols - 2):
                    image1[i, j] = np.sum((m1 * image[i - 2:i + 3, j - 2:j + 3, k]))

        image1 = cv2.convertScaleAbs(image1)
        cv2.imwrite('img/EdgeImgs/EdgePic.jpg', image1)
        cv2.waitKey(0)

    def EdgeCanny(self):
        self.picColor = cv2.imread(self.pic_path, 1).astype(np.uint8) #彩色模式读取
        # 2. 灰度转换
        blur = cv2.cvtColor(self.picColor, cv2.COLOR_BGR2GRAY)
        # 3. 求x，y方向的Sobel算子
        gradx = cv2.Sobel(blur, cv2.CV_16SC1, 1, 0)
        grady = cv2.Sobel(blur, cv2.CV_16SC1, 0, 1)
        # 4. 使用Canny函数处理图像，x,y分别是3求出来的梯度，低阈值50，高阈值150
        edge_output = cv2.Canny(gradx, grady, 50, 150)

        cv2.imwrite('img/EdgeImgs/EdgePic.jpg', edge_output)
        cv2.waitKey(0)

    def HoughDetect(self, threshold, color):
        img = cv2.GaussianBlur(self.cpic, (3, 3), 0)
        edges = cv2.Canny(img, 50, 150, apertureSize=3)

        lines = cv2.HoughLines(edges, 1, np.pi / 2, threshold)
        result = img.copy()

        if lines is None:
            cv2.imwrite('img/EdgeImgs/EdgePic.jpg', result)
            return

        for i_line in lines:
            for line in i_line:
                rho = line[0]
                theta = line[1]
                if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
                    pt1 = (int(rho / np.cos(theta)), 0)
                    pt2 = (int((rho - result.shape[0] * np.sin(theta)) / np.cos(theta)), result.shape[0])
                    if color == 'red':
                        cv2.line(result, pt1, pt2, (0, 0, 255))
                    if color == 'white':
                        cv2.line(result, pt1, pt2, (255, 255, 255))
                    if color == 'black':
                        cv2.line(result, pt1, pt2, (0, 0, 0))
                    if color == 'green':
                        cv2.line(result, pt1, pt2, (0, 255, 0))
                    if color == 'blue':
                        cv2.line(result, pt1, pt2, (0, 255, 0))
                else:
                    pt1 = (0, int(rho / np.sin(theta)))
                    pt2 = (result.shape[1], int((rho - result.shape[1] * np.cos(theta)) / np.sin(theta)))
                    if color == 'red':
                        cv2.line(result, pt1, pt2, (0, 0, 255), 1)
                    if color == 'white':
                        cv2.line(result, pt1, pt2, (255, 255, 255), 1)
                    if color == 'black':
                        cv2.line(result, pt1, pt2, (0, 0, 0), 1)
                    if color == 'green':
                        cv2.line(result, pt1, pt2, (0, 255, 0), 1)
                    if color == 'blue':
                        cv2.line(result, pt1, pt2, (0, 255, 0), 1)

        cv2.imwrite('img/EdgeImgs/EdgePic.jpg', result)
        cv2.waitKey(0)

    def HoughDetectP(self, threshold, MinLineLength, MaxLineGap, color):
        img = cv2.GaussianBlur(self.cpic, (3, 3), 0)
        edges = cv2.Canny(img, 50, 150, apertureSize=3)
        linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold,  minLineLength = MinLineLength,maxLineGap = MaxLineGap)

        result_P = img.copy()

        if linesP is None:
            cv2.imwrite('img/EdgeImgs/EdgePic.jpg', result_P)
            return

        for i_P in linesP:
            for x1, y1, x2, y2 in i_P:
                if color == 'red':
                    cv2.line(result_P, (x1, y1), (x2, y2), (0, 0, 255), 3)
                if color == 'white':
                    cv2.line(result_P, (x1, y1), (x2, y2), (255, 255, 255), 3)
                if color == 'black':
                    cv2.line(result_P, (x1, y1), (x2, y2), (0, 0, 0), 3)
                if color == 'green':
                    cv2.line(result_P, (x1, y1), (x2, y2), (0, 255, 0), 3)
                if color == 'blue':
                    cv2.line(result_P, (x1, y1), (x2, y2), (255, 0, 0), 3)

        cv2.imwrite('img/EdgeImgs/EdgePic.jpg', result_P)
        cv2.waitKey(0)