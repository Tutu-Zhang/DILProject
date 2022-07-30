import math
import cv2
import numpy as np
import struct
import matplotlib.pyplot as plt

class ColorMenu:
    #读取彩色图片
    def __init__(self, pic_path):
        self.pic_path = pic_path
        self.pic = cv2.imread(self.pic_path, 1).astype(np.uint8)
        self.b = self.pic[:, :, 0]
        self.g = self.pic[:, :, 1]
        self.r = self.pic[:, :, 2]

        hsv = cv2.cvtColor(self.pic, cv2.COLOR_BGR2HSV)
        self.h = hsv[:, :, 0]
        self.s = hsv[:, :, 1]
        self.v = hsv[:, :, 2]

    def ShowPic(self):
        cv2.imshow('当前图片', self.pic)
        cv2.waitKey(0)  # 0一直显示，直到有键盘输入。也可以是其他数字.


    #RGB
    def Bchannel(self):
        cv2.imwrite('img/colorImgs/Cchannel.jpg', self.b)
       # cv2.imshow('Bchannel', self.b)
        cv2.waitKey(0)
    def Gchannel(self):
        cv2.imwrite('img/colorImgs/Cchannel.jpg', self.g)
        #cv2.imshow('Gchannel', self.g)
        cv2.waitKey(0)
    def Rchannel(self):
        cv2.imwrite('img/colorImgs/Cchannel.jpg', self.r)
        #cv2.imshow('Gchannel', self.r)
        cv2.waitKey(0)

    #HSV
    def Hchannel(self):
        cv2.imwrite('img/colorImgs/Cchannel.jpg', self.h)
        #cv2.imshow('Hchannel', self.h)
        cv2.waitKey(0)
    def Schannel(self):
        cv2.imwrite('img/colorImgs/Cchannel.jpg', self.s)
        #cv2.imshow('Schannel', self.s)
        cv2.waitKey(0)
    def Vchannel(self):
        cv2.imwrite('img/colorImgs/Cchannel.jpg', self.v)
        #cv2.imshow('Vchannel', self.v)
        cv2.waitKey(0)