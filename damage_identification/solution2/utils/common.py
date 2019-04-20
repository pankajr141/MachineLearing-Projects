'''
Created on Jul 1, 2016

@author: pankajrawat
'''
from image import operations as imop
import matplotlib.cm as cm
import numpy as np
import os
import cv2


def getColorMappedImage2(image, colormap):
    median = cv2.medianBlur(image,5)
    median_grey = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    th, dst = cv2.threshold(median, 90, 255, cv2.THRESH_TOZERO)
    dst_grey = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(dst_grey, median_grey)
    th, diff = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV);
    return diff
    
    
def getColorMappedImage3(image, colormap, mode="predict"):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    colormap2 = cm.get_cmap(colormap)
    image = colormap2(image)
    image = image[:,:,:-1]
    image = image * 255
    image = image.astype(np.uint8)
    return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   

def getColorMappedImage(image, colormap, mode="predict"):

    return getColorMappedImage3(image, colormap, mode="predict")  
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #image = cv2.GaussianBlur(image,(5,5),0)
    #image[image < 100] = 0   

    colormap = cm.get_cmap("jet")
    image = colormap(image)
    #imop.display(image, wait=None, label="del")
    #imop.display(getResizedImageScaled(image, 1200, 800), wait=None, label="del")
    """
    image[:, :, 1] = 0 
    image[:, :, 0] = 0 
    image[:, :, 3] = 0
    """
    threshold = 0.85 if mode == "predict" else 1
    threshold = 0.85
    image = image.astype(np.uint8)
    image = image * 255
    #image[image < 0.75] = 0
    image[image < threshold] = 0
    return image[:, :, 2]

def getResizedImageScaled(image, xscale, yscale=None):
    if not yscale:
        yscale = xscale
    dimention = (xscale, yscale)
    resized = cv2.resize(image, dimention, interpolation = cv2.INTER_AREA)
    return resized


def getResizedImageRGB(image, newDimentionPixels):
    height, width, colors = image.shape
    if height > width:
        ratioNewToOld = newDimentionPixels / float(height)
        dimention = (int(width*ratioNewToOld), newDimentionPixels) 
    else:
        ratioNewToOld = newDimentionPixels / float(width)
        dimention = (newDimentionPixels, int(height*ratioNewToOld)) 

    # perform the actual resizing of the image and show it
    resized = cv2.resize(image, dimention, interpolation = cv2.INTER_AREA)
    if height > width:
        mask = np.zeros((newDimentionPixels, newDimentionPixels - dimention[0], 3),np.uint8)
        resized = np.concatenate((resized, mask), axis=1)
    else:
        mask = np.zeros((newDimentionPixels - dimention[1], newDimentionPixels, 3),np.uint8)
        resized = np.concatenate((resized, mask), axis=0)
    return resized

def getAvgRGBValues(image):
    meanValues = int(np.mean(image[:,:,0])), int(np.mean(image[:,:,1])), int(np.mean(image[:,:,2]))
    meanValue = np.array(meanValues).reshape(1, 3)
    return meanValue

def getwhitePercent(image):
    whiteSize = image[image > 140].size
    imageSize = image.size
    percentWhite = int((float(whiteSize)/imageSize)*100)
    return percentWhite

def getPixelDensity(image):
    return int(np.mean(image[:,:]))

def sliding_window_l(image, stepSize, windowSize):
   for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def additionalFeatures(image):
    grid = 10
    image = getResizedImageScaled(image, 128)
    winH = image.shape[0]/grid
    percents = []
    pixelDensitys = []
    pixelDensitysB = []
    pixelDensitysG = []
    pixelDensitysR = []
    #imop.display(image)
    for (x, y, window) in sliding_window_l(image, stepSize=winH, windowSize=(winH, winH)):
        if window.shape[0] != winH or window.shape[1] != winH:
            continue
        percent = getwhitePercent(window)
        pixelDensity = getAvgRGBValues(window) #getPixelDensity(window)
        pixelDensitysB.append(pixelDensity[0][0])
        pixelDensitysG.append(pixelDensity[0][1])
        pixelDensitysR.append(pixelDensity[0][2])
        #print pixelDensity
        percents.append(percent)
        pixelDensitys.append(pixelDensity)
    
    overall = getwhitePercent(image)
    percents.append(overall)
    percents.extend(pixelDensitys)
    
    """NEW"""
    overall = getAvgRGBValues(image)
    percents = pixelDensitysB
    percents.extend(pixelDensitysG)
    percents.extend(pixelDensitysR)
    percents.append(overall[0][0])
    percents.append(overall[0][1])
    percents.append(overall[0][2])
    """ """
    percents = np.array(percents).reshape(1, (grid * grid * 3) + 3)  

    #percents = np.array(percents).reshape(1, (grid * grid * 2) + 1)  
    return percents

def getImageColorTone(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print image.shape
    avg = np.mean(image)
    if avg < 110:
        return "Dark"
    return "Light"

def IgnorePatch(image):
    tone  = getAvgRGBValues(image)
    print tone
    if tone[0][1] > 180:
        return True
    if tone[0][2] > 180:
        return True