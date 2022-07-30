import math
import cv2
import numpy as np
import struct
import matplotlib.pyplot as plt

class geometryMenu:
    # 读取彩色图片
    def __init__(self, pic_path):
        self.pic_path = pic_path
        self.pic = cv2.imread(self.pic_path, 1).astype(np.uint8)

    def ShowPic(self):
        cv2.imshow('当前图片', self.pic)
        cv2.waitKey(0)  # 0一直显示，直到有键盘输入。也可以是其他数字.

    #二维移动,向右x,向下y
    def ImageMove(self, x, y):
        height, width, channel = self.pic.shape
        M = np.float32([[1, 0, x], [0, 1, y]])
        self.moved_pic = cv2.warpAffine(self.pic, M, (width, height))
        cv2.imwrite('img/geometryImgs/moved_pic.jpg',self.moved_pic)
        cv2.waitKey(0)

    def ImageResizeTimes(self, x_times, y_times):
        self.moved_pic = cv2.resize(self.pic, (0, 0), fx=x_times, fy=y_times)
        cv2.imwrite('img/geometryImgs/moved_pic.jpg',self.moved_pic)
        cv2.waitKey(0)

    def ImageResizePixel(self, x_p, y_p):
        self.moved_pic = cv2.resize(self.pic, (x_p, y_p))
        cv2.imwrite('img/geometryImgs/moved_pic.jpg',self.moved_pic)
        cv2.waitKey(0)

    def HorizentalFlip(self):
        self.moved_pic = cv2.flip(self.pic, 1, dst=None)
        cv2.imwrite('img/geometryImgs/moved_pic.jpg',self.moved_pic)
        cv2.waitKey(0)

    def VerticalFlip(self):
        self.moved_pic = cv2.flip(self.pic, 0, dst=None)
        cv2.imwrite('img/geometryImgs/moved_pic.jpg', self.moved_pic)
        cv2.waitKey(0)

    def CrossFlip(self):
        self.moved_pic = cv2.flip(self.pic, -1, dst=None)
        cv2.imwrite('img/geometryImgs/moved_pic.jpg', self.moved_pic)
        cv2.waitKey(0)

    def FreeRotate(self, angle, size):
        height, width, channel = self.pic.shape
        S = cv2.getRotationMatrix2D((width/2, height/2), angle, size)
        self.moved_pic = cv2.warpAffine(self.pic, S, (width, height))
        cv2.imwrite('img/geometryImgs/moved_pic.jpg', self.moved_pic)
        cv2.waitKey(0)

    def FixedRotate(self, mode):
        if mode == '90':
            self.moved_pic = cv2.rotate(self.pic, cv2.ROTATE_90_CLOCKWISE)
        elif mode == '180':
            self.moved_pic = cv2.rotate(self.pic, cv2.ROTATE_180)
        elif mode == '270':
            self.moved_pic = cv2.rotate(self.pic, cv2.ROTATE_90_COUNTERCLOCKWISE)

        cv2.imwrite('img/geometryImgs/moved_pic.jpg', self.moved_pic)
        cv2.waitKey(0)










