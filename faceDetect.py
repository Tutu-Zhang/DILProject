import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
from imutils import face_utils

class FaceDetect:
    def __init__(self, pic_path):
        self.pic_path = pic_path

        if pic_path != "":
            self.pic = cv2.imread(self.pic_path, 1).astype(np.uint8)


    def ShowPic(self):
        cv2.imshow('当前图片', self.pic)
        cv2.waitKey(0)  # 0一直显示，直到有键盘输入。也可以是其他数字.

    def detectFace(self):
        im = np.float32(self.pic) / 255.0
        # Calculate gradient
        gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=1)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        self.facePic = self.pic

        face_detect = dlib.get_frontal_face_detector()
        rects = face_detect(self.pic, 1)
        for (i, rect) in enumerate(rects):
            (x, y, w, h) = face_utils.rect_to_bb(rect)
            self.facePic = cv2.rectangle(self.pic, (x, y), (x + w, y + h), (255, 255, 255), 3)

        cv2.imwrite("img/faceImg/facePic.jpg", self.facePic)
        cv2.waitKey(0)  # 0一直显示，直到有键盘输入。也可以是其他数字.

    def videoDetect(self):
        video_capture = cv2.VideoCapture(0)
        face_detect = dlib.get_frontal_face_detector()
        flag = 0

        while True:

            ret, frame = video_capture.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = face_detect(gray, 1)

            for (i, rect) in enumerate(rects):
                (x, y, w, h) = face_utils.rect_to_bb(rect)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow('Face detect, Press ESC Quit', frame)

            if cv2.waitKey(3) is 27:
                break
            if cv2.getWindowProperty('Face detect, Press ESC Quit', cv2.WND_PROP_AUTOSIZE) < 1:
                break

        video_capture.release()
        cv2.destroyWindow(self)
