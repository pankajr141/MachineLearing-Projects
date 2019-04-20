'''
Created on May 12, 2016

@author: pankajrawat
'''
import cProfile
import os

import cv2
import sys
from datetime import datetime
from image import preprocess_image
import common.constants as const

from image import operations as imop
import feature_creation
from sklearn.externals import joblib


def reshapeWindow():
    global window_width, window_height
    window_width = 500
    window_height = 600


def main():
    start = datetime.now()
    reshapeWindow()
    cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('dst_rt', window_width, window_height)

    imageFile = os.path.join("D:\New Volume\Documents\HailWingProject\HailDamageGoogleNoMetaData", '1-28-1.jpg')
    img = imop.readBinaryImage(imageFile)
    imgRGB = imop.readImage(imageFile)

    """ Preprocess image (smoothning, morphological transformation, threshold)"""
    img = preprocess_image.getProcessedImage(img)
    imop.display(img)
    
    tmpImagePath = os.path.join("dataset", "tmpimg")
    if not os.path.exists(tmpImagePath) or 1==1:
        img = preprocess_image.repairImage(img)
        img = preprocess_image.breakImage(img)
        joblib.dump(img, tmpImagePath)
    img = joblib.load(tmpImagePath)


    print "TIME Taken",  datetime.now() - start

	''' During TRAIN_MODE, CSV is generated with manual user input as Yes or No.
	More details present in identifyDamages function '''
    if const.TRAIN_MODE:
        feature_creation.identifyDamages(img)
        sys.exit()

    img = feature_creation.markDamages(img, imgRGB)

    print "TIME Taken",  datetime.now() - start
    cv2.imwrite(const.OUTPUT_IMAGE_PATH, imgRGB)

    imop.display(imgRGB)
    imop.display(img)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    main()
    pr.disable()
    pr.print_stats(sort='time')
