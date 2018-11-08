# -*- coding: utf-8 -*-
import cv2 as cv
import numpy
import glob
ethalonImg = 'D:\im1.jpg'
a = cv.imread(ethalonImg)
a.shape[:2]
cv.imshow('a',a)
list1 = []
for filename in glob.glob('D:\CVPR_IMG\*.jpg'):
    list1.append(filename)
for file in list1:
    a=cv.imread(file)
    cv.imshow('a',a)
    cv.waitKey()