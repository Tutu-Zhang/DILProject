import math
import cv2
import numpy as np
import struct
import matplotlib.pyplot as plt
import random

class FilterMenu:
    def __init__(self, pic_path):
        self.pic_path = pic_path
        self.pic = cv2.imread(self.pic_path, 1).astype(np.uint8)
        self.greypic = cv2.imread(self.pic_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

    def ShowPic(self):
        cv2.imshow('当前图片', self.pic)
        cv2.waitKey(0)  # 0一直显示，直到有键盘输入。也可以是其他数字.

    def GaussNoise(self, average, variance):
        # 将图片的像素值归一化，存入矩阵中
        image = np.array(self.pic / 255, dtype=float)
        # 生成正态分布的噪声，其中0表示均值，0.1表示方差
        noise = np.random.normal(average, variance, image.shape)
        # 将噪声叠加到图片上
        out = image + noise
        # 将图像的归一化像素值控制在0和1之间，防止噪声越界
        out = np.clip(out, 0.0, 1.0)
        # 将图像的像素值恢复到0到255之间
        out = np.uint8(out * 255)

        cv2.imwrite('img/filterImgs/FilterPic.jpg', out)
        cv2.waitKey(0)

    def PepperSaltyNoise(self, threshold):

        # 待输出的图片
        output = np.zeros(self.pic.shape, np.uint8)
        # 椒盐噪声的阈值
        prob = threshold
        thres = 1 - prob
        # 遍历图像，获取叠加噪声后的图像
        for i in range(self.pic.shape[0]):
            for j in range(self.pic.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    # 添加胡椒噪声
                    output[i][j] = 0
                elif rdn > thres:
                    # 添加食盐噪声
                    output[i][j] = 255
                else:
                    # 不添加噪声
                    output[i][j] = self.pic[i][j]

        cv2.imwrite('img/filterImgs/FilterPic.jpg', output)
        cv2.waitKey(0)

    def gaussFilter(self, core):
        num = core

        grayImage = cv2.GaussianBlur(self.pic, (num, num), 0)

        cv2.imwrite('img/filterImgs/FilterPic.jpg', grayImage)
        cv2.waitKey(0)

    def AverageFilter(self):
        output = np.zeros(self.greypic.shape, np.uint8)
        # 遍历图像，进行均值滤波
        for i in range(self.greypic.shape[0]):
            for j in range(self.greypic.shape[1]):
                # 滤波器内像素值的和
                sum = 0
                # 遍历滤波器内的像素值
                for m in range(-1, 2):
                    for n in range(-1, 2):
                        # 防止越界
                        if 0 <= i + m < self.greypic.shape[0] and 0 <= j + n < self.greypic.shape[1]:
                            # 像素值求和
                            sum += self.greypic[i + m][j + n]
                            # 求均值，作为最终的像素值
                output[i][j] = int(sum / 9)

        cv2.imwrite('img/filterImgs/FilterPic.jpg', output)
        cv2.waitKey(0)

    # 获取列表的中间值的函数
    def get_middle(self, array):
        # 列表的长度
        length = len(array)
        # 对列表进行选择排序，获得有序的列表
        for i in range(length):
            for j in range(i + 1, length):
                # 选择最大的值
                if array[j] > array[i]:
                    # 交换位置
                    temp = array[j]
                    array[j] = array[i]
                    array[i] = temp
        return array[int(length / 2)]

    def middleFilter(self):
        output = np.zeros(self.greypic.shape, np.uint8)
        # 存储滤波器范围内的像素值
        self.array = []
        # 遍历图像，进行中值滤波
        for i in range(self.greypic.shape[0]):
            for j in range(self.greypic.shape[1]):
                # 清空滤波器内的像素值
                self.array.clear()
                # 遍历滤波器内的像素
                for m in range(-1, 2):
                    for n in range(-1, 2):
                        # 防止越界
                        if 0 <= i + m < self.greypic.shape[0] and 0 <= j + n < self.greypic.shape[1]:
                            # 像素值加到列表中
                            self.array.append(self.greypic[i + m][j + n])
                            # 求中值，作为最终的像素值
                output[i][j] = self.get_middle(self.array)

        cv2.imwrite('img/filterImgs/FilterPic.jpg', output)
        cv2.waitKey(0)

    def LeastFilter(self):
        output = np.zeros(self.greypic.shape, np.uint8)
        ######### Begin #########
        for i in range(self.greypic.shape[0]):
            for j in range(self.greypic.shape[1]):
                # 最小值滤波器
                minone = 0
                for m in range(-1, 2):
                    for n in range(-1, 2):
                        if 0 <= i + m < self.greypic.shape[0] and 0 <= j + n < self.greypic.shape[1]:
                            # 通过比较判断是否需要更新最小值
                            if self.greypic[i + m][j + n] < minone:
                                minone = self.greypic[i + m][j + n]

                # 更新最小值
                output[i][j] = minone

        cv2.imwrite('img/filterImgs/FilterPic.jpg', output)
        cv2.waitKey(0)

    def LargestFilter(self):
        output = np.zeros(self.greypic.shape, np.uint8)
        ######### Begin #########
        for i in range(self.greypic.shape[0]):
            for j in range(self.greypic.shape[1]):
                # 最大值滤波器
                maxone = 0
                for m in range(-1, 2):
                    for n in range(-1, 2):
                        if 0 <= i + m < self.greypic.shape[0] and 0 <= j + n < self.greypic.shape[1]:
                            # 通过比较判断是否需要更新最大值
                            if self.greypic[i + m][j + n] > maxone:
                                maxone = self.greypic[i + m][j + n]

                # 更新最小值
                output[i][j] = maxone

        cv2.imwrite('img/filterImgs/FilterPic.jpg', output)
        cv2.waitKey(0)

    def PassFilter(self, rangemin, rangemax, filt_num):
        # 待输出的图片
        output = np.zeros(self.greypic.shape, np.uint8)
        # 遍历图像，进行均值滤波
        array = []
        # 带通的范围
        min = rangemin
        max = rangemax
        for i in range(self.greypic.shape[0]):
            for j in range(self.greypic.shape[1]):
                # 滤波器内像素值的和
                array.clear()
                if min < self.greypic[i][j] < max:
                    output[i][j] = self.greypic[i][j]
                else:
                    if filt_num == '0':
                        output[i][j] = 0
                    elif filt_num == '255':
                        output[i][j] = 255

        cv2.imwrite('img/filterImgs/FilterPic.jpg', output)
        cv2.waitKey(0)

    def BlockFilter(self, rangemin, rangemax, filt_num):
        # 待输出的图片
        output = np.zeros(self.greypic.shape, np.uint8)
        # 遍历图像，进行均值滤波
        array = []
        # 带阻的范围
        min = rangemin
        max = rangemax
        for i in range(self.greypic.shape[0]):
            for j in range(self.greypic.shape[1]):
                # 滤波器内像素值的和
                array.clear()
                if min < self.greypic[i][j] < max:
                    if filt_num == '0':
                        output[i][j] = 0
                    elif filt_num == '255':
                        output[i][j] = 255
                else:
                    output[i][j] = self.greypic[i][j]

        cv2.imwrite('img/filterImgs/FilterPic.jpg', output)
        cv2.waitKey(0)