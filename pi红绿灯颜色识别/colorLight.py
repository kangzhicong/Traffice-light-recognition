#coding:utf-8
'''
    根据颜色进行红绿灯分割
    需要库: opencv-python
    sudo pip3 install opencv-python
'''
import os
import sys
import cv2
import numpy as np


def img_subtract(a1,a2):
    a1 = a1.astype(np.float32)
    a2 = a2.astype(np.float32)
    s = a1 - a2
    s = np.where(s < 0 ,0 ,s)
    return s

def cnt_area(cnt):
    """返回轮廓的面积"""
    area = cv2.contourArea(cnt)
    return area


class LightRecog(object):
    def __init__(self):
        self.dict_thresh = {"rg":80,"br":50,"gb":100} 


    '''
        需要调用的函数
        返回两个参数: img , (c1,c2)
        判断，if c1 > c2，绿灯; c1 < c2 ，红灯;  其他: 没有识别出
    '''
    def recoglight(self , img,flag = 0):
        b,g,r = cv2.split(img)

        #red
        subtracted = img_subtract(r,g) 
        result = np.where(subtracted > self.dict_thresh["rg"] , 255,0)
        result = result.astype(np.uint8)
        contours_r, hierarchy = cv2.findContours(result,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours_r.sort(key=cnt_area, reverse=True) #排序



        #green
        subtracted = img_subtract(g,b) 
        result = np.where(subtracted > self.dict_thresh["gb"] , 255,0)
        result = result.astype(np.uint8)
        contours_g, hierarchy = cv2.findContours(result,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours_g.sort(key=cnt_area, reverse=True) #排序

        if len(contours_r) > 0:
            cv2.drawContours(img,contours_r,-1,(255,255,0),-1)#画轮廓
        if (len(contours_g) > 0):
            cv2.drawContours(img,contours_g,-1,(255,0,255),-1)#画轮廓

        pixels_r = 0
        for c in contours_r:
            pixels_r += len(c)
        pixels_g = 0
        for c in contours_g:
            pixels_g += len(c)
        
        conf_g = pixels_g/(0.001 + pixels_g + pixels_r)
        conf_r = pixels_r/(0.001 + pixels_g + pixels_r)

        print(pixels_g , pixels_r)
        print("绿灯概率:%3f, 红灯概率:%3f\n"%(conf_g , conf_r))
        return  img,(conf_g,conf_r)


    '''
        识别颜色
        输入:图像、识别的类别
        输出:结果图像，用来可视化显示（见test_show函数）
    '''
    def recog_color(self, img , flag):

        b,g,r = cv2.split(img)
        if flag == -1:
            return img,False
        elif flag == 0:#red 
            # print("red")
            subtracted = img_subtract(r,g) 
            result = np.where(subtracted > self.dict_thresh["rg"] , 255,0)
            result = result.astype(np.uint8)
            contours, hierarchy = cv2.findContours(result,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            contours.sort(key=cnt_area, reverse=True) #排序
            print(len(contours))
            if len(contours) == 0:
                return img,False
            else:
            # if cnt_area(contours[0]) > self.dict_thresh["area"]:
                cv2.drawContours(img,contours,-1,(0,0,255),-1)#画轮廓
            # else:
                # return img,False

        elif flag == 1:#green
            # print("green")
            # subtracted = img_subtract(g,r) 
            # result = np.where(subtracted > self.dict_thresh["gr"] , 255,0)
            subtracted = img_subtract(g,b) 
            result = np.where(subtracted > self.dict_thresh["gb"] , 255,0)
            result = result.astype(np.uint8)
            contours, hierarchy = cv2.findContours(result,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            contours.sort(key=cnt_area, reverse=True) #排序
            print(len(contours))
            if len(contours) == 0:
                return img,False
            # if cnt_area(contours[0]) > self.dict_thresh["area"]:
            else:
                cv2.drawContours(img,contours,-1,(0,255,0),-1)#画轮廓
            # else:
                # return img,False
        return img,True


def test_show():
    ColorRecog = LightRecog()
    for i in range(1,6):
        img_path = "green/%d.jpg"%i
        test_img = cv2.imread(img_path)
        test_img = cv2.resize(test_img,(640,480))
        result_img , _ = ColorRecog.recog_color(test_img , flag = 1)
        cv2.imshow("s" , result_img)
        cv2.waitKey(0)

    for i in range(1,6):
        img_path = "red/%d.jpg"%i
        test_img = cv2.imread(img_path)
        test_img = cv2.resize(test_img,(640,480))
        result_img , _ = ColorRecog.recog_color(test_img , flag = 0)
        cv2.imshow("s" , result_img)
        cv2.waitKey(0)
    

    cap = cv2.VideoCapture('green.avi')
    while True:
        ret,frame = cap.read()
        if ret == False:
            break
        frame = cv2.resize(frame,(640,480))
        result_img , _ = ColorRecog.recog_color(frame , flag = 1)
        cv2.imshow("s" , result_img)
        cv2.waitKey(0)
    cap.release()

    cap = cv2.VideoCapture('red.avi')
    while True:
        ret,frame = cap.read()
        if ret == False:
            break
        frame = cv2.resize(frame,(640,480))
        result_img , _ = ColorRecog.recog_color(frame , flag = 0)
        cv2.imshow("s" , result_img)
        cv2.waitKey(0)
    cap.release()


if __name__ == "__main__":
    ColorRecog = LightRecog()
    for i in range(1,6):
        img_path = "green/%d.jpg"%i
        test_img = cv2.imread(img_path)
        test_img = cv2.resize(test_img,(640,480))
        result_img , _ = ColorRecog.recoglight(test_img , flag = 1)
        cv2.imshow("s" , result_img)
        cv2.waitKey(0)

    for i in range(1,6):
        img_path = "red/%d.jpg"%i
        test_img = cv2.imread(img_path)
        test_img = cv2.resize(test_img,(640,480))
        result_img , _ = ColorRecog.recoglight(test_img , flag = 0)
        cv2.imshow("s" , result_img)
        cv2.waitKey(0)
    

    cap = cv2.VideoCapture('green.avi')
    while True:
        ret,frame = cap.read()
        if ret == False:
            break
        frame = cv2.resize(frame,(640,480))
        result_img , _ = ColorRecog.recoglight(frame , flag = 1)
        cv2.imshow("s" , result_img)
        cv2.waitKey(0)
    cap.release()

    cap = cv2.VideoCapture('red.avi')
    while True:
        ret,frame = cap.read()
        if ret == False:
            break
        frame = cv2.resize(frame,(640,480))
        result_img , _ = ColorRecog.recoglight(frame , flag = 0)
        cv2.imshow("s" , result_img)
        cv2.waitKey(0)
    cap.release()