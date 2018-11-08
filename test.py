# -*- coding: utf-8 -*-
from __future__ import print_function

import glob
import itertools as it
import math
import os
import time

import cv2 as cv
import numpy as np

from multiprocessing.pool import ThreadPool
# local modules

def affine_skew(tilt, phi, img, mask=None):
    '''
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai

    Ai - is an affine transform matrix from skew_img to img
    '''
    h, w = img.shape[:2]
    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255
    A = np.float32([[1, 0, 0], [0, 1, 0]])
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c,-s], [ s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32( np.dot(corners, A.T) )
        x, y, w, h = cv.boundingRect(tcorners.reshape(1,-1,2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv.warpAffine(img, A, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REPLICATE)
    if tilt != 1.0:
        s = 0.8*np.sqrt(tilt*tilt-1)
        img = cv.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv.resize(img, (0, 0), fx=1.0/tilt, fy=1.0, interpolation=cv.INTER_NEAREST)
        A[0] /= tilt
    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv.warpAffine(mask, A, (w, h), flags=cv.INTER_NEAREST)
    Ai = cv.invertAffineTransform(A)
    return img, mask, Ai


def affine_detect(detector, img, mask=None, pool=None):
    '''
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs

    Apply a set of affine transformations to the image, detect keypoints and
    reproject them into initial image coordinates.
    See http://www.ipol.im/pub/algo/my_affine_sift/ for the details.

    ThreadPool object may be passed to speedup the computation.
    '''
    params = [(1.0, 0.0)]
    for t in 2**(0.5*np.arange(1,6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple( np.dot(Ai, (x, y, 1)) )
        if descrs is None:
            descrs = []
        return keypoints, descrs

    keypoints, descrs = [], []
    if pool is None:
        ires = it.imap(f, params)
    else:
        ires = pool.imap(f, params)

    for i, (k, d) in enumerate(ires):
        print('affine sampling: %d / %d\r' % (i+1, len(params)), end='')
        keypoints.extend(k)
        descrs.extend(d)

    print()
    return keypoints, np.array(descrs)

def matchCenter(img, min, max):
    hsv_min = np.array(min, np.uint8)
    hsv_max = np.array(max, np.uint8)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    thresh = cv.inRange(hsv, hsv_min, hsv_max)
    moments = cv.moments(thresh, 1)
    dM01 = moments['m01']
    dM10 = moments['m10']
    dArea = moments['m00']
    if dArea > 100:
        x = int(dM10 / dArea)
        y = int(dM01 / dArea)
        return x, y
    return 0, 0


def kaze_match(kps1, descs1, gray1, im2_path, f, min, max):
    start_time = time.time()
    im2 = cv.imread(im2_path)
    gray2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

    detector = cv.xfeatures2d_SIFT.create()
    #kps2, descs2 = detector.detectAndCompute(gray2, None)
    kps2, descs2= affine_detect(detector,gray2,pool = None)
    bf = cv.BFMatcher(cv.NORM_L1,crossCheck = False)
    matches = bf.knnMatch(descs1, descs2,k=2)
    good = []
    center = [0, 0]
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
                good.append([m])
                center[0] += kps2[m.trainIdx].pt[0]
                center[1] += kps2[m.trainIdx].pt[1]
    matchTime = time.time() - start_time

    info = "img: {}, mathes: {}, good: {}, delta: {}, time/size: {} sec" \
            .format(im2_path, len(matches), len(good), len(matches) - len(good),
                    matchTime / os.path.getsize(im2_path) * 1000000)

    if min != None and max != None:
        x, y = matchCenter(im2, min, max)
        # похибка локалізації (відстань між реальним розміщенням предмета в кадрі та розпізнаним)
        locError = math.sqrt((x - center[0] / len(good)) ** 2 + (y - center[1] / len(good)) ** 2)
        info += ', localization error: {}'.format(locError)

    f.write(info + '\n')
    print(info)
        # im3 = cv.drawMatchesKnn(gray1, kps1, gray2, kps2, good, None, flags=2)
        # cv.namedWindow("output", cv.WINDOW_NORMAL)  # Create window with freedom of dimensions
        # imS = cv.resize(im3, (960, 540))  # Resize image
        # cv.imshow("output", imS)


def processAll(list, ethalonImg, min, max):

    im1 = cv.imread(ethalonImg)
    gray1 = cv.cvtColor(im1, cv.COLOR_RGB2GRAY)
    detector = cv.xfeatures2d_SIFT.create()
    #kps1, descs1 = detector.detectAndCompute(gray1, None)
    kps1, descs1 = affine_detect(detector, gray1, pool = None)
    print("1 \n")
    for file in list:
        kaze_match(kps1, descs1, gray1, file, f, min, max)
        key = cv.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            break
if __name__ == '__main__':
    imageList = []
    ethalonImg = 'D:\im1.JPG'
   # for filename in glob.glob('D:\CVPR_IMG\*.jpg'):
           # print("2 \n")
            #imageList.append(filename)
    f = open("result.txt", "a")
    imageList1 = ['D:\CVPR_IMG\P81103-185045.JPG']
    min = (0, 0, 153)
    max = (255, 51, 255)
    f.write('\n' + "=======police bobik=======" + '\n')
    f.write('\n' + "=======images with obj=======" + '\n')
    processAll(imageList1, ethalonImg, min, max)
    '''
    f.write("=======images without obj=======" + '\n')
    processAll(imageListWithoutObj, ethalonImg, min, max)
    min = (0, 0, 153)
    max = (255, 51, 255)
    f.write('\n' + "=======remote control=======" + '\n')
    processAll(imageListRemote, ethalonImgRemote)
    '''
    f.close()
    cv.destroyAllWindows()