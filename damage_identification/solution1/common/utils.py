'''
Created on May 25, 2016

@author: pankajrawat
'''
import math
import constants as const
import numpy as np

def getAngleDifference(angle1, angle2):
    return 180 - abs(abs(angle1 - angle2) - 180)


def getAngle(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    rads = math.atan2(-dy,dx)
    rads %= 2*math.pi
    degs = math.degrees(rads)
    return degs


def debug(msg, level=1):
    if not const.DEBUG:
        return
    print "DEBUG >> ", msg
    
    
def getCentroid(contour):
    import cv2
    M = cv2.moments(contour)
    if M['m00'] == 0:
        centroid_x = int(math.ceil(np.median(contour[:, :, 0], axis=0)))
        centroid_y = int(math.ceil(np.median(contour[:, :, 1], axis=0)))
        return centroid_x, centroid_y
    centroid_x = int(M['m10']/M['m00'])
    centroid_y = int(M['m01']/M['m00'])
    return centroid_x, centroid_y
