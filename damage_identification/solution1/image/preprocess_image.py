'''
Created on May 27, 2016

@author: pankajrawat
'''
import cv2
import math
import numpy as np
import traceback
import copy
from common import utils
from common import constants as const
from image import operations as imop
from common import geometry as geo

"""
Function preprocess image by applying some basic image processing techniques to emerge the dark regions
Common preprocessing techniques include smoothning, thresholding, morphological transformation etc.
"""
def getProcessedImage(img):
    #img = createHistographEqualization(img)

    """ Smoothing the image to remove noise """
    #img = cv2.bilateralFilter(img,9,75,75)
    #img = cv2.blur(img,(5,5))
    
    img = cv2.GaussianBlur(img,(5,5),0)

    imop.display(img)
    """ Applying morphological transformation """
    kernel = np.ones((15,15), np.uint8)
    #img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel) - img
    img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel) 
    #img[img > 230] = 255p

    # Try to use other transformation, like closing etc, may yield better output
    #img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel) 
    #img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
    #img = img - cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    #img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    """ Applying threshold """
    img[img < 35] = 0
    #img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel) 
    # Filtering not done, required to preprocess noise , use probability and connected component based filtering
    return img



"""
Function modifies input image by appending the points along the direction, 
used to complete lines which are breaking due to erosion or other cv operation
"""
def _extendLine(img, pts, angleInclination, direction, imgLineColor):
    if direction == "L":
        pts = pts[pts[:,1].argsort()]
    elif direction == "R":
        pts = pts[pts[:,1].argsort()][::-1]
    elif direction == "T":
        pts = pts[pts[:,0].argsort()]      
    elif direction == "B":
        pts = pts[pts[:,0].argsort()][::-1]

    x1, y1 = pts[0][1], pts[0][0]


    """ Find all the expected points along extended line """
    extendedLineCoords = []
    for n in range(10): # 10
        x2, y2 = geo.getNewCoordsAlongDirection(direction, angleInclination, (x1, y1), n)
        if not any([x2, y2]):
            continue
        """Change logic to calculate parallel line"""
        if direction in ['L', 'R']:
            # Point Above
            extendedLineCoords.append((y2-1, x2))
            extendedLineCoords.append((y2  , x2))
            # Point Below
            extendedLineCoords.append((y2+1, x2))
        else:
            # Point Left
            extendedLineCoords.append((y2, x2-1))
            extendedLineCoords.append((y2, x2))
            # Point Right
            extendedLineCoords.append((y2, x2+1))
    
    if const.DISPLAY_COLOR:
        height, width, _ = img.shape
    else:
        height, width = img.shape

    """Removing points which lie outside image"""
    extendedLineCoords = filter(lambda x: x[0] >= 0 and x[1] >= 0 and x[0] < height and x[1] < width, extendedLineCoords)

    #print img.shape
    joinUpto = 0    # Location up to which need to join in extendedLineCoords
    pointsFound = 0    
    for cntr, extendedLineCoord in enumerate(extendedLineCoords):
        joinUpto += 1
        """ We will skip immediate points while checking for point existence, because immediate points may be of current contour,
         Here 3 is point, abovePoint and belowPoint """
        if cntr < const.REPAIR_IMMEDIATE_PIXEL_SKIP * 3:
            continue
        
        colorFlag = False

        if const.DISPLAY_COLOR and not any(img[extendedLineCoord]) > 0:
            continue
        elif not const.DISPLAY_COLOR and not img[extendedLineCoord] > 0:
            continue
        
        pointsFound += 1
        if pointsFound == const.REPAIR_MIN_OTHER_COUNTER_POINT_REQUIRED_FOR_JOINING:            
            break

    if not pointsFound:
        return

    for cntr, (y2, x2) in enumerate(extendedLineCoords[0: joinUpto + 1]):
        lineColor = const.BLUE if const.DISPLAY_COLOR else imgLineColor
        img[y2, x2] = lineColor
    #print img[img[:,:,0] > 0]
    #print img[(10, 100), (11, 11)]
    #print img[extendedLineCoords]
    #print img[extendedLineCoords.astype(int)[:, 0]]


"""
Function returns true if their is a line in input image along the direction
"""
def _getAngleofLine(img, pts, direction):
    # Sorted pixels by x coords
    if direction == "L":
        pts = pts[pts[:,1].argsort()]
    elif direction == "R":
        pts = pts[pts[:,1].argsort()][::-1]
    elif direction == "T":
        pts = pts[pts[:,0].argsort()]
    elif direction == "B":
        pts = pts[pts[:,0].argsort()][::-1]

    # Find all pixels having coords > CONST_SKIP from extremepoint
    x1, y1 = pts[0][1], pts[0][0]
    #===========================================================================
    # # Check if midpoint of 
    # startPixels = pts[np.where(pts[:, 1] == pts[0][1])]
    # startPixels = startPixels[startPixels[:, 0].argsort()]
    # midLocation = math.ceil(len(startPixels)/2)
    # y1, x1, i1 = startPixels[midLocation]
    # print "Midpoint", y1, x1, i1 
    #===========================================================================

    lineSegmentPassed = 0
    lineSegmentError = 0
    prevAngleInclination = None
    sumAngleInclination = 0
    imgLineColor = []
    for _ in range(const.REPAIR_MAX_PIXEL_POINT_JUMP):
        try:
            if const.DISPLAY_COLOR:
                imgLineColor.append(img[y1, x1, 0])
            else:
                imgLineColor.append(img[y1, x1])
                
            if all([const.DISPLAY_COLOR, const.REPAIR_FLAG_DISPLAY_LINE_MIDPOINS]): 
                cv2.circle(img, (x1, y1), 0, (0, 0, 255), -1)
            if direction == 'L':
                pixelSpecifiedDistance = pts[np.where(pts[:, 1] == x1 + const.REPAIR_PIXEL_JUMP_WHILE_DETECTING_LINE)]
            elif direction == 'R':
                pixelSpecifiedDistance = pts[np.where(pts[:, 1] == x1 - const.REPAIR_PIXEL_JUMP_WHILE_DETECTING_LINE)]
            elif direction == 'T':
                pixelSpecifiedDistance = pts[np.where(pts[:, 0] == y1 + const.REPAIR_PIXEL_JUMP_WHILE_DETECTING_LINE)]
            elif direction == 'B':
                pixelSpecifiedDistance = pts[np.where(pts[:, 0] == y1 - const.REPAIR_PIXEL_JUMP_WHILE_DETECTING_LINE)]

            if direction in ['L', 'R']:
                pixelSpecifiedDistance = pixelSpecifiedDistance[pixelSpecifiedDistance[:, 0].argsort()]
            else:
                pixelSpecifiedDistance = pixelSpecifiedDistance[pixelSpecifiedDistance[:, 1].argsort()]

            midLocation = math.ceil(len(pixelSpecifiedDistance)/2)
            
            if const.DISPLAY_COLOR:
                y2, x2, i2 = pixelSpecifiedDistance[midLocation]
            else:
                y2, x2 = pixelSpecifiedDistance[midLocation]

            utils.debug("Line segment: (x1, y1) => (x2, y2) %s %s => %s %s" % (x1, y1, x2, y2), 2)
            angleInclination = utils.getAngle(x1, y1, x2, y2)

            x1, y1 = x2, y2
            if prevAngleInclination == None:
                prevAngleInclination = angleInclination
                continue
            angleDifference = utils.getAngleDifference(angleInclination, prevAngleInclination)
            if direction == 'L' and angleInclination > 180:
                angleInclination = angleInclination - 360
                
            utils.debug(" $$ PrevAngle %s || Angle %s || Difference %s" % (prevAngleInclination, angleInclination, angleDifference), 2)

            if abs(angleDifference) > const.REPAIR_MAX_ANGLE_DIFFERENCE_ALLOWED:
                lineSegmentError += 1

            #sumAngleInclination = sumAngleInclination if abs(angleDifference) > 20 else sumAngleInclination + angleDifference
            sumAngleInclination = sumAngleInclination if abs(angleDifference) > const.REPAIR_MAX_ANGLE_DIFFERENCE_ALLOWED else sumAngleInclination + angleInclination
            lineSegmentPassed += 1
            prevAngleInclination = angleInclination
        except Exception, err:
            #print err, type(err)
            break

    if not lineSegmentPassed:
        return (False, None, None)

    # Check if more than 20% of segments have angle more then threshold and atleast 5 segment passed
    if not lineSegmentPassed or lineSegmentError/lineSegmentPassed > const.REPAIR_MAX_LINE_ERROR or lineSegmentPassed < const.REPAIR_MIN_SEGMENT_PASS:
        return (False, None, None)

    avgAngleInclination = sumAngleInclination/ float(lineSegmentPassed - lineSegmentError)

    # Adjusting negative angles to real values
    if direction in ['L']:
        avgAngleInclination = avgAngleInclination if avgAngleInclination > 0 else 360 + avgAngleInclination

    imgLineColor =  np.sum(imgLineColor) / len(imgLineColor)
    utils.debug("%s Summary %s" % ('*'* 30, '*'* 30, ))
    utils.debug("DIR: %s, sumAngleInclination: %s, AvgAngleInclination: %s" % (direction, sumAngleInclination, avgAngleInclination))

    #print "*" * 20, "Summary:", "*" * 20, "\nDIR:", dir, "\nsumAngleInclination:", sumAngleInclination, "\nAvgAngleInclination:", avgAngleInclination
    utils.debug("lineSegmentPassed: %s, lineSegmentError: %s" % (lineSegmentPassed, lineSegmentError))
    utils.debug("*" * 70)
    return (True, avgAngleInclination, imgLineColor)


"""
Function will repair missing line segments lost during thresholding operation
"""
def repairImage(img):
    minAreaThreshold = const.REPAIR_MIN_AREA_REQUIRED
    if const.DISPLAY_COLOR and not imop.isColored(img):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cntImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if imop.isColored(img) else copy.copy(img)
    _, contours, _ = cv2.findContours(cntImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    
    print "Repair: Number of components", len(contours)
    utils.debug("Repair: Total contours previous repair => %d " % (len(contours)), 1)
    contours_ = filter(lambda x: minAreaThreshold <= cv2.contourArea(x), contours)


    """ Extend Contours to recover missing pixels """
    for cntr, contour in enumerate(contours_):
        try:
            #if not cntr in range(139, 140):
            #    continue
            #print "----------------- CNTR", cntr
            #cv2.drawContours(img, contours_, cntr, (0, 255, 0), thickness=1)
            mask = np.zeros(img.shape, np.uint8)
            cv2.drawContours(mask,[contour],0,255,-1)
            pts = np.where(mask > 0)
            #pts = np.nonzero(mask)
            pts = np.transpose(pts)

            """ Code to detect angleOfInclination and extend the lines based on that"""
            for direction in ['L', 'R', 'T', 'B']:
                (isLine, angleInclination, imgLineColor) = _getAngleofLine(img, pts, direction)
                if not isLine:
                    continue
                _extendLine(img, pts, angleInclination, direction, imgLineColor)
        except Exception, err:
            traceback.print_exc()
            print err
    return img

""" 
Function objective is to separate regions from input image which are connected and thus will go undetected
Regions include blobs, rectangles.
As of now only blob separation is implemented
"""
def breakImage(img):

    # Use copy image findCounters tends to modify original image    
    if const.DISPLAY_COLOR and not imop.isColored(img):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cntImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if imop.isColored(img) else copy.copy(img)

    components, markers, stats, centroids = cv2.connectedComponentsWithStats(cntImg, 8, cv2.CV_32S)
    print "Break: Number of components", components
    #markers = cv2.watershed(img, markers)

    overallBlobs = []
    for cntr in range(1, components):
        try:
            #if not cntr in range(4, 5):
            #    continue                
            #print "----------------- CNTR", cntr
            pixelsArea = stats[cntr][4]
            if pixelsArea < const.BREAK_MIN_AREA_REQUIRED:
                continue
            mask = np.zeros(img.shape, np.uint8)
            mask[markers == cntr] = const.GREEN if const.DISPLAY_COLOR else 255
            blobs = _detectBlobs(img, mask)

            if not blobs:
                continue

            if not overallBlobs:
                overallBlobs = blobs
                continue

            overallBlobs.extend(blobs)
        except Exception, err:
            traceback.print_exc()
            print err
    img = _markBlobs(img, overallBlobs)
    return img

"""
Function will separate out block regions from rest of image by removing the pixels along the radius
"""
def _markBlobs(img, blobs):
    for blob in blobs:
        try:
            centroid_x, centroid_y, radius = blob
            color = const.ORANGE if const.DISPLAY_COLOR else (0)
            cv2.circle(img, (centroid_x, centroid_y), radius, color, 2)
        except Exception, err:
            traceback.print_exc()
            print err
    return img

"""
Function returns identified blob regions in the image, mask
This function accept segments and returns detected blob points 
Algo: 
1. Apply Distance transformation to remove pale regions
2. Find contours of remaining and check if they are blobs
"""

def _detectBlobs(img, mask):
    if const.DISPLAY_COLOR:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    dist, _ = cv2.distanceTransformWithLabels(mask, cv2.DIST_L2, 3)#cv2.DIST_MASK_PRECISE)
    
    min, max = dist.min(), dist.max()
    if not max:
        return

    dist = ( 1 * (dist - min) )/ (max - min)
    percentile =  np.percentile(dist[dist > 0.0].flatten(), const.BREAK_PERCENTILE_BLOB)
    
    # If even 1 point of 97 percent lies above quantile region
    if percentile >= const.BREAK_PERCENTILE_THRESHOLD:
        return

    dist[dist < percentile] = 0
    dist[dist >= percentile] = 255

    kernel = np.ones((5,5), np.uint8)
    dist = cv2.morphologyEx(dist, cv2.MORPH_DILATE, kernel)
    dist = dist.astype(np.uint8)
    
    #imop.display(dist, label='blob1', wait=False)
    
    blobPoints = getBlobPoints(dist)

    _, contours, _ = cv2.findContours(copy.copy(dist), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    baseRadius = percentile * (max - min) + min
    blobs = []
    for contour in contours:
        blobPoint = filter(lambda x: cv2.pointPolygonTest(contour, x.pt, measureDist=False) > 0, blobPoints)
        if not blobPoint:
            continue
        centroid_x, centroid_y = map(lambda x: int(round(x)), blobPoint[0].pt)
        radius = int(round(baseRadius + blobPoint[0].size)) - 1
        blobs.append([centroid_x, centroid_y, radius])
    return blobs

def getBlobPoints(blobImg):
    #blobImg = cv2.cvtColor(blobImg, cv2.COLOR_GRAY2RGB)
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
     
    # Change thresholds
    #params.minThreshold = 10;
    #params.maxThreshold = 200;
     
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1
     
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.4
     
    # Filter by Convexity
    
    params.filterByConvexity = True
    params.minConvexity = 0.0

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.2
    
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(255 - blobImg)
    """
    im_with_keypoints = cv2.drawKeypoints(blobImg, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    imop.display(im_with_keypoints, label='blob2', wait=False)
    """
    return keypoints
