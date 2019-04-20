'''
Created on Jun 1, 2016

@author: pankajrawat
'''
import cv2
import numpy as np
import pandas as pd
from image import operations as imop
import common.constants as const
from sklearn.externals import joblib
import os
import copy
import traceback


def _getFeatures(img, markers, stats, centroids, cntr, imageDimention):
    #if not cntr in range(872, 873):
    #    continue   
    
    #===========================================================================
    # pixelsArea = len(img[markers == cntr])
    #===========================================================================
    
    mask = np.zeros(img.shape, np.uint8)
    mask[markers == cntr] = 255

    #===========================================================================
    # xcoords, ycoords = np.where(mask == 255)
    # yMin = np.min(xcoords)
    # yMax = np.max(xcoords)
    # xMin = np.min(ycoords)
    # xMax = np.max(ycoords)
    # width = xMax - xMin + 1
    # height = yMax - yMin + 1
    #===========================================================================

    #xMin, yMin, width, height = cv2.boundingRect(contour)  # Check Performance which is better
    #print centroids[cntr]
    #centroid_x, centroid_y = centroids[cntr]

    width = stats[2]
    height = stats[3]
    pixelsArea = stats[4]
    xMin = stats[0]
    yMin = stats[1]
    xMax = xMin + width 
    yMax = yMin + height
    #print xMin, yMin, xMax, yMax, width, height
    
    #imop.display(mask)
    
    # leftmost = tuple(contour[contour[:,:,0].argmin()][0])
    # rightmost = tuple(contour[contour[:,:,0].argmax()][0])
    # topmost = tuple(contour[contour[:,:,1].argmin()][0])
    # bottommost = tuple(contour[contour[:,:,1].argmax()][0])
    # print "ExtremePoints: ", "L:", leftmost, "R:", rightmost, "T:", topmost, "B:", bottommost
            
    #imgCounter = np.zeros((height, width, 1), np.uint8)
    imgComponent = mask[yMin: yMax, xMin: xMax]
    #cv2.drawContours(imgCounter, [contour], 0, [255], thickness=-1)
    
    rectArea = width * height
    if rectArea == 0:
        extent = 0
    else:
        extent = float(pixelsArea)/rectArea
    
    #===========================================================================
    # hull = cv2.convexHull(imgCounter)
    # hullArea = cv2.contourArea(hull)
    # if hullArea == 0:
    #     solidity = 0
    # else:
    #     solidity = float(pixelsArea)/hullArea
    #===========================================================================

    #Mean Intensity greyscale, better use originar greyscale
    #mean_val = cv2.mean(im,mask = mask)
    #===========================================================================
    # 
    # print "Area:", pixelsArea
    # print "Extent: ", extent #, ", Solidity:", solidity
    # print "Orignal Shape: ", imgComponent.shape
    #===========================================================================
        
    resized = imop.getResizedImage(imgComponent, imageDimention)
    #imop.display(resized, wait=False, label="resized", time=1)
    return (width, height, pixelsArea, extent, resized)
    
'''
Function is used to generate training dataset / images.  [MANUALLY]

First it will generate a connected components array, then for each component will ask user if it is a damage, 
if answer is yes then it will extract that region create featured based on above functions 
and save it on disk as CSV, for later processing during model training. 
'''

def identifyDamages(img):
    if const.DISPLAY_COLOR:
        print "ERROR: Training not allowed in colored mode, please set DISPLAY_COLOR=FALSE in configuration file"
        import sys
        sys.exit(1)

    if not imop.isColored(img):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cntImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    components, markers, stats, centroids = cv2.connectedComponentsWithStats(cntImg, 8, cv2.CV_32S)
    print "Potential Damages: Number of components", components
    #markers = cv2.watershed(img, markers)

    for cntr in range(1, components):
        try:
            print "################  CNTR: ", cntr
            img[markers == cntr] = const.GREEN
            imop.display(img, wait=False, time=1)
			
			#Generating features which would be used for training.
            features = _getFeatures(cntImg, markers, stats[cntr], centroids[cntr], cntr, const.IMAGE_DIMENTION)
            width, height, pixelsArea, extent, pixels = features
            
            if pixelsArea <= const.TRAINING_AREA_THRESHOLD:
                print 'Threshold met skipping ...........'
                img[markers == cntr] = const.WHITE
                continue      
             
            input = raw_input("Is Defect: ") 
            if input.strip() == "":
                print 'skipping on behalf of user ...........'
                img[markers == cntr] = const.WHITE
                continue         
            
            isDefect = True if input == 'y' else False
            
            cols = (['defect', 'width', 'height', 'pixelsArea', 'extent'])
            cols.extend(map(lambda x: 'px_' + str(x) , range(const.IMAGE_DIMENTION ** 2)))
            data = pixels.flatten()
            data = np.append(np.array([isDefect, width, height, pixelsArea, extent]), data)
			
            df = pd.DataFrame([data], columns=cols)
            outputFile = const.TRAIN_OUTPUT_FILE
            header = True if not os.path.exists(outputFile) else False
            df.to_csv(outputFile, index=False, mode='a', header=header)
            color = const.RED if isDefect else const.YELLOW
            img[markers == cntr] = color
        except Exception, err:
            print "ERROR:", err
            print traceback.print_exc()

''' Function is used to mark connected component as damaged or not based on ML model trained using TRAIN_MODE'''
def markDamages(img, imgRGB):
    if not imop.isColored(img):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if const.DISPLAY_COLOR:
        img[np.where((img == const.BLUE).all(axis = 2))] =  const.GREY
        img[np.where((img == const.ORANGE).all(axis = 2))] =  const.BLACK
    
    cntImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if imop.isColored(img) else copy.copy(img)
    
    components, markers, stats, centroids = cv2.connectedComponentsWithStats(cntImg, 8, cv2.CV_32S)

    print "Potential Damages: Number of components", components
    #markers = cv2.watershed(img, markers)

    estimator = joblib.load(const.ESTIMATER_PICKLE_FILE)
    for cntr in range(1, components):
        try:
            print "--------------  CNTR: ", cntr

            #imop.display(img, wait=False, time=1)
            imop.display(imgRGB, wait=False, time=1)
            
            features = _getFeatures(cntImg, markers, stats[cntr], centroids[cntr], cntr, const.IMAGE_DIMENTION)
            width, height, pixelsArea, extent, pixels = features
            data = pixels.flatten()
            data = np.append(np.array([width, height,  pixelsArea, extent]), data)
            isDefect = estimator.predict([data])
            print "isDefect:", isDefect
            if isDefect:
                img[markers == cntr] = const.RED
                width = stats[cntr][2]
                height = stats[cntr][3]
                radius = width if width < height else height
                if radius < 5:
                    continue
                cv2.circle(imgRGB, (int(centroids[cntr][0]), int(centroids[cntr][1])), radius, (0, 0, 255), 1)
                #imgRGB[markers == cntr] = [0, 0,  255, 0]                
            else:
                centroids[cntr]
                color = const.RED if const.DISPLAY_COLOR else 255
                img[markers == cntr] = [0, 255, 255]
        except Exception, err:
            print "ERROR:", err
            print traceback.print_exc()
    return img
	

#===============================================================================
# def createFeatures(contour, trainMode, estimator=None):
#     try:
#         xMin, yMin = contour.min(axis=0)[0]
#         xMax, yMax = contour.max(axis=0)[0]
#         width = xMax - xMin + 1
#         height = yMax - yMin + 1
#         
#         #xMin, yMin, width, height = cv2.boundingRect(contour)  # Check Performance which is better
#         #print xMin, yMin, xMax, yMax
#     
#         # leftmost = tuple(contour[contour[:,:,0].argmin()][0])
#         # rightmost = tuple(contour[contour[:,:,0].argmax()][0])
#         # topmost = tuple(contour[contour[:,:,1].argmin()][0])
#         # bottommost = tuple(contour[contour[:,:,1].argmax()][0])
#         # print "ExtremePoints: ", "L:", leftmost, "R:", rightmost, "T:", topmost, "B:", bottommost
#         
#         contour[:, :, 1] = contour[:, :, 1] - yMin
#         contour[:, :, 0] = contour[:, :, 0] - xMin
#     
#         imgCounter = np.zeros((height, width, 1), np.uint8)
#         cv2.drawContours(imgCounter, [contour], 0, [255], thickness=-1)
#     
#         M = cv2.moments(contour)
#         if M['m00'] == 0:
#             centroid_x = centroid_y = 0
#         else:
#             centroid_x = int(M['m10']/M['m00'])
#             centroid_y = int(M['m01']/M['m00'])
#         area = cv2.contourArea(contour)
#         rectArea = width * height
#         if rectArea == 0:
#             extent = 0
#         else:
#             extent = float(area)/rectArea
#         
#         hull = cv2.convexHull(contour)
#         hullArea = cv2.contourArea(hull)
#         if hullArea == 0:
#             solidity = 0
#         else:
#             solidity = float(area)/hullArea
#     
#         #===========================================================================
#         # mask = np.zeros(imgCounter.shape,np.uint8)
#         # cv2.drawContours(mask,[contour],0,255,-1)
#         # pixelPoints = np.transpose(np.nonzero(mask))
#         #===========================================================================
#     
#         #Mean Intensity greyscale, better use originar greyscale
#         #mean_val = cv2.mean(im,mask = mask)
#         
#         print "Centroid_X: %s, Centroid_Y %s" % (centroid_x, centroid_y)
#         print "Area:", area, ", isConvex:", cv2.isContourConvex(contour)    
#         print "Extent: ", extent, ", Solidity:", solidity
#         #print "Arc Length", cv2.arcLength(contour)
#         #cv2.circle(imgCounter, (centroid_x, centroid_y), 0, (0, 0, 255), 3)
#         print "Orignal Shape: ", imgCounter.shape
#         
#         #imop.display(imgCounter, wait=False)
#         
#         newDimentionPixels = 32
#         resized = imop.getResizedImage(imgCounter, newDimentionPixels)
#         #print df
#         imop.display(resized, wait=False, label="resized", time=1)
# 
#         
#         if trainMode:
#             input = raw_input("Is Defect: ") if area >= 5 else 'nonexist'            
#             isDefect = True if input == 'y' else False
#             
#             cols = (['defect', 'width', 'height', 'Cx', 'Cy', 'area', 'isConvex', 'extent', 'solidity'])
#             cols.extend(map(lambda x: 'px_' + str(x) , range(newDimentionPixels ** 2)))
#             
#             data = resized.flatten()
#             data = np.append(np.array([isDefect, width, height, centroid_x, centroid_y, area, cv2.isContourConvex(contour), extent, solidity]), data)
#             
#             df = pd.DataFrame([data], columns=cols)
#             outputFile = const.TRAIN_OUTPUT_FILE
#             header = True if not os.path.exists(outputFile) else False
#             df.to_csv(outputFile, index=False, mode='a', header=header)
#         else:
#             data = resized.flatten()
#             data = np.append(np.array([width, height, centroid_x, centroid_y, area, cv2.isContourConvex(contour), extent, solidity]), data)
#             isDefect = estimator.predict(data)
#         return isDefect
#     except Exception, err:
#         print err
#===============================================================================