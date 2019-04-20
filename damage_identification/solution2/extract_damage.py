'''
Created on Jun 30, 2016

@author: pankajrawat
'''
import cv2
import os
import time
import cProfile
import numpy as np
import random
import matplotlib.cm as cm
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.externals import joblib


''' Imported from solution 1'''
from image import operations as imop
from damage.utils import common


pickle_dir = "D:\\New Volume\\workspace\\classifiers"
damageDir = "D:\\New Volume\\Documents\\Damages\\sysdamages"

def getRandomName():
    return "IMG_" + str(random.randint(1, 1000000000000)) + ".jpg"

def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    stepSize = int(stepSize/3)
    stepSize = int(windowSize[0]/3)
    for y in xrange(0, image.shape[0], stepSize):
        for x in xrange(0, image.shape[1], stepSize):
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def identifyDamages(imageFile, outFile, outFileCombined, display=False):

    clf = joblib.load(os.path.join(pickle_dir, "SVC"))   
    #clf = joblib.load(os.path.join(pickle_dir, "RandomForestClassifier"))

    image = cv2.imread(imageFile)

    imageOriginal = image.copy()
    imageDumping = image.copy()
    imageOriginal2 = image.copy()
	
    image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    
    #image = cv2.GaussianBlur(image,(5,5),0)
    #image = cv2.medianBlur(image,7,0)

    if display:
        #imop.display(common.getResizedImageScaled(image, 1200, 800), label="BNW", wait=None) 
        im_display = 255 - common.getColorMappedImage(image, "hsv_r")         
        imop.display(common.getResizedImageScaled(im_display, 1200, 800), label="BNW", wait=None)

    clone = np.zeros(image.shape, np.uint8)
    window_size = [20, 28, 60, 80, 128, 140]

	'Using Different window sizes'
    if image.shape[0] > 1700:
        window_size = filter(lambda x: x > 100, window_size)
    elif image.shape[0] < 800:
        window_size = filter(lambda x: x < 60, window_size)
    elif image.shape[0] < 400:
        window_size = filter(lambda x: x < 40, window_size)
    else:
        window_size = filter(lambda x: x >=28 and x < 100, window_size)

    print window_size

    for windowsSize in window_size:
        print "WindowsSize :>> ", windowsSize
        (winW, winH) = (windowsSize, windowsSize)
        for (x, y, window) in sliding_window(image, stepSize=32, windowSize=(winW, winH)):
        #for (x, y, window) in sliding_window(common.getColorMappedImage(image, "hsv_r", mode="predict"), stepSize=32, windowSize=(winW, winH)):

            # if the window does not meet our desired window size, ignore it
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            segmentImage = common.getResizedImageScaled(window, 64)

            hist = np.array(cv2.calcHist([cv2.cvtColor(segmentImage, cv2.COLOR_BGR2GRAY)],[0], None,[100],[0,256])).reshape(1, 100)
            segmentImage = common.getColorMappedImage(segmentImage, "hsv_r", mode="predict")
            #segmentImage = cv2.GaussianBlur(segmentImage,(3,3),0)
            segmentImage = 255 - segmentImage
            features = common.additionalFeatures(segmentImage)
            #meanValues = common.additionalFeatures(segmentImage)
            orig = imageDumping[y:y + winH, x:x + winW]
            orig = common.getResizedImageScaled(orig, 64)

            segmentImageShow = segmentImage.copy()

            if display:
                displayImage = imageOriginal.copy()
                cv2.rectangle(displayImage, (x, y), (x + winW, y + winH), (0, 255, 255), 2)
                imop.display(common.getResizedImageScaled(displayImage, 1600, 1000), label="movingDamageIdentificationWindow_g", wait=None)    

            #segmentImage = common.getResizedImageScaled(segmentImage, 24)
            segmentImage = segmentImage.reshape(1, segmentImage.size)
            segmentImage = np.concatenate((segmentImage, features), axis=1)
            #segmentImage = np.concatenate((segmentImage, features, hist), axis=1)
            segmentImage = np.concatenate((features, hist), axis=1)
            segmentImage  = features

            proba = clf.predict_proba(segmentImage)
            damageIdentified = clf.predict(segmentImage)
            if display:
                imop.display(orig, label='ORG', wait=False)
                segmentImageShow[segmentImageShow<=140] = 0
                imop.display(segmentImageShow, label='TIF', wait=False)
                cv2.waitKey(1)
           
            if not damageIdentified:
                continue

            if proba[0][1] < 0.80:
                continue

            #imop.display(orig, label='ORG', wait=False)
            #imop.display(segmentImageShow, label='TIF', wait=True)
            cv2.rectangle(imageOriginal, (x, y), (x + winW, y + winH), (0, 0, 255), 2)
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 0, 255), 2)

            
            if not os.path.exists(damageDir):
                os.makedirs(damageDir)
            DPATH = os.path.join(damageDir, getRandomName())
            cv2.imwrite(DPATH, orig)

    clone = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
    _, contours, _ = cv2.findContours(clone.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(imageOriginal2, contours, -1, (0, 0, 255), thickness=1)
    cv2.imwrite(outFileCombined, imageOriginal2);

    cv2.imwrite(outFile, imageOriginal);
    cv2.waitKey(0)

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    directory = "D:\New Volume\Documents\HailDamageGoogleNoMetaData"
    """
    for file in os.listdir(directory):
        start = datetime.now()
        imageFile = os.path.join(directory, file)
        print "File: " + imageFile
        outFile = os.path.join("imageset", "G-" + file)
        outFileCombined = os.path.join("imageset", "CG-" + file)
        identifyDamages(imageFile, outFile, outFileCombined)
        print datetime.now() - start
    """
    imageFile = "sample.jpg"
    outFile =  os.path.join("imageset", "G-sample.jpg")
    outFileCombined  = os.path.join("imageset", "COM-sample.jpg")
    identifyDamages(imageFile, outFile, outFileCombined, display=False)
    
    pr.disable()
    #pr.print_stats(sort='time')
