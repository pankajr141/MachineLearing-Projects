'''
Created on Jun 30, 2016

@author: pankajrawat
'''
import os
import cv2
import math
import numpy as np
from image import operations as imop
from damage import extract_damage as ed
import matplotlib as mpl
import matplotlib.cm as cm
#from matplotlib.pylab import cm
import matplotlib.pyplot as plt
from damage.utils import common

"""
pltImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(pltImage, interpolation='nearest')
plt.waitforbuttonpress()
plt.savefig('data.png')
#exit()
"""

#print cm.cmaps_listed

def stack(listImages):
    print len(listImages)
    numRows = len(listImages) / 5
    rows = math.ceil(numRows)
    data = None
    for image in listImages:
        data = np.concatenate([data, image])
    
    
def matplotColorMaps():
    global image
    image = cv2.GaussianBlur(image,(3,3),0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    print cm.cmaps_listed
    print cm.cmap_d
    print cm.colors
    
    ol = []
    #colormaps = ['inferno_r', 'plasma_r', 'viridis_r', 'viridis', 'magma', 'inferno', 'magma_r', 'plasma']

    colormaps = cm.cmaps_listed.keys()
    colormaps = cm.cmap_d.keys()
    #colormaps = map(lambda x: cm.get_cmap(x), cm.cmaps_listed.keys())
    #colormaps = map(lambda x: cm.get_cmap(x), cm.cmap_d.keys())

    for index, colormap in enumerate(colormaps):
        #plt.subplot(2, 4, index + 1)     
        plt.subplot(8, 20, index + 1)
        plt.title(colormap, fontsize=8)
        plt.axis('off')
        colormap = cm.get_cmap(colormap)
        plt.imshow(image, cmap=colormap, interpolation='nearest')
    plt.waitforbuttonpress()   
    plt.savefig("colormaps.jpg")
    #cv2.imshow("list", np.hstack(ol))
    #imop.display(image)


def openCVColorMaps():
    lst = []
    lst.append(image)
    print image.shape
    formats  = [
                cv2.COLOR_BGR2HSV, 
                cv2.COLOR_BGR2YUV,
                cv2.COLOR_BGR2HLS,
                cv2.COLOR_BGR2LAB,
                cv2.COLOR_BGR2Luv,            
                ]
    
    for format in formats:
        out = cv2.cvtColor(image, format)
        print out.shape
        lst.append(out)
    
    cv2.imshow("crop", np.hstack(lst))
    
    image = cv2.GaussianBlur(image,(3,3),0)
    imop.display(image)



def test(imagePath):
    print imagePath
    basePath = os.path.basename(imagePath)
    image = cv2.imread(imagePath)
    
    #image = np.load("npsaveimage-hd8.npy")
    image = cv2.fastNlMeansDenoisingColored(image,None,10,10,7,21)
    #np.save("npsaveimage-hd7", image)
    
    #imop.display(image)
    
    orig = image.copy()
    image = cv2.GaussianBlur(image,(3,3),0)

    #imageTrain = common.getColorMappedImage(image, "jet", mode="train")
    image = common.getColorMappedImage(image, "hsv_r", mode="predict")
    image = 255 - image

    #imop.display(common.getResizedImageScaled(image, 1200, 800), wait=None)
    #image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,91,1)
    #print common.additionalFeatures(image)
    """
    for colormap2 in cm.cmap_d.keys():
        print colormap2
        colormap2 = cm.get_cmap(colormap2)
        image1 = colormap2(image)
        #image1 = cv2.GaussianBlur(image1,(5,5),0)
        image1 = image1[:,:,:-1]
        image1 = image1 * 255
        image1 = image1.astype(np.uint8)
        image1 = cv2.cvtColor(255 - image1, cv2.COLOR_BGR2GRAY)     
        imop.display(common.getResizedImageScaled(image1, 1200, 800), label="Real")
        
    image = common.getColorMappedImage(image, "bwr_r", mode="predict")

    """
    image[image < 140] = 0

    orig = common.getResizedImageScaled(orig, 1200, 800)
    image = common.getResizedImageScaled(image, 1200, 800)
    
    #imageTrain = common.getResizedImageScaled(imageTrain, 1200, 800)

    """
    morphName = "morph-" + basePath
    trainName = "train-" + basePath
    cv2.imwrite(morphName, image*255)
    cv2.imwrite(trainName, imageTrain*255)
    """

    imop.display(image, label="Morphed", wait=True)
    #imop.display(imageTrain, label="Train", wait=None)
    #imop.display(orig, label="originala", wait=True)
    cv2.waitKey(4)

if __name__ == "__main__":
    fileName = os.path.join("IMG_753933016.jpg")
    image = cv2.imread(fileName)
    print image
    #hog = cv2.HOGDescriptor()
    #h = hog.compute(image)
    #print type(h), h.shape

    basePath = "D:\\New Volume\\Documents\\Damages\\Mdamages_3"
    for cntr, path in enumerate(os.listdir(basePath)):
        print cntr
        fileName = os.path.join(basePath, path)
        test(fileName)