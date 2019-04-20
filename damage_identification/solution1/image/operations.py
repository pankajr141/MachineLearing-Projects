'''
Created on May 27, 2016

@author: pankajrawat
'''

import cv2
import numpy as np
import common.constants as const


def readBinaryImage(imageFile):
    return cv2.cvtColor(cv2.imread(imageFile), cv2.COLOR_BGR2GRAY)

def readImage(imageFile):
    return cv2.cvtColor(cv2.imread(imageFile), 0)


def createHistographEqualization(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #img = clahe.apply(img)
    return cv2.equalizeHist(img)

def display (img, wait=True, label='dst_rt', time=None):
    cv2.imshow(label, img)
    if not any([wait, time]):
        return
    if not wait and time:
        cv2.waitKey(time)
        return
    cv2.waitKey(0)

def getResizedImage(imgCounter, newDimentionPixels):
    height, width = imgCounter.shape
    if height > width:
        ratioNewToOld = newDimentionPixels / float(height)
        dimention = (int(width*ratioNewToOld), newDimentionPixels) 
    else:
        ratioNewToOld = newDimentionPixels / float(width)
        dimention = (newDimentionPixels, int(height*ratioNewToOld)) 

    # perform the actual resizing of the image and show it
    resized = cv2.resize(imgCounter, dimention, interpolation = cv2.INTER_AREA)
    if height > width:
        mask = np.zeros((newDimentionPixels, newDimentionPixels - dimention[0]),np.uint8)
        resized = np.concatenate((resized, mask), axis=1)
    else:
        mask = np.zeros((newDimentionPixels - dimention[1], newDimentionPixels),np.uint8)
        resized = np.concatenate((resized, mask), axis=0)
    return resized

def isColored(img):
    if len(img.shape) < 3:
        return False
    return True