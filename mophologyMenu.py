import math
import cv2
import numpy as np
import struct
import matplotlib.pyplot as plt

class mophologyMenu:

    def __init__(self, pic_path):
        self.pic_path = pic_path
        self.pic = cv2.imread(self.pic_path, 1).astype(np.uint8)

    def ShowPic(self):
        cv2.imshow('当前图片', self.pic)
        cv2.waitKey(0)  # 0一直显示，直到有键盘输入。也可以是其他数字.

    def Erode(self, constructSize):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (constructSize, constructSize), (-1, -1))
        erosion = cv2.erode(self.pic, kernel)
        cv2.imwrite('img/mophologyImgs/mophoPic.jpg',erosion)
        cv2.waitKey(0)

    def Dilate(self, constructSize):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (constructSize, constructSize))
        dilation = cv2.dilate(self.pic, kernel)
        cv2.imwrite('img/mophologyImgs/mophoPic.jpg', dilation)
        cv2.waitKey(0)

    def Open(self, cons):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (cons, cons))
        Open = cv2.morphologyEx(self.pic, cv2.MORPH_OPEN, kernel)
        cv2.imwrite('img/mophologyImgs/mophoPic.jpg', Open)
        cv2.waitKey(0)

    def Close(self, cons):
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (cons, cons))
        Close = cv2.morphologyEx(self.pic, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite('img/mophologyImgs/mophoPic.jpg', Close)
        cv2.waitKey(0)