'''
Created on Jun 2, 2016

@author: pankajrawat
'''

import math
def getNewCoordsAlongDirection(direction, angleInclination, position, n):
    """ Y coordinates starts from top left """
    x1, y1 = position
    x2, y2 = None, None
    if direction == 'L':
        # Sign X (correct) Y (may change)
        if 0 <= angleInclination <= 90:
            x2 = int(math.floor(x1 - n * math.cos(math.radians(angleInclination))))
            y2 = int(math.floor(y1 - n * math.sin(math.radians(angleInclination))))
        elif  270 <= angleInclination < 360:
            x2 = int(math.floor(x1 - n * math.cos(math.radians(angleInclination))))
            y2 = int(math.floor(y1 + n * math.sin(math.radians(angleInclination))))
    elif direction == 'R':
        # Sign X (correct) Y (may change)
        if 90 <= angleInclination <= 180:
            x2 = int(math.floor(x1 - n * math.cos(math.radians(angleInclination))))
            y2 = int(math.floor(y1 + n * math.sin(math.radians(angleInclination))))
        elif  180 < angleInclination <= 270:
            x2 = int(math.floor(x1 - n * math.cos(math.radians(angleInclination))))
            y2 = int(math.floor(y1 + n * math.sin(math.radians(angleInclination))))
    elif direction == 'T':
        # Sign Y (correct) X (may change)
        if  180 < angleInclination <= 270:
            x2 = int(math.floor(x1 - n * math.cos(math.radians(angleInclination))))
            y2 = int(math.floor(y1 + n * math.sin(math.radians(angleInclination))))
        elif  270 < angleInclination <= 360:
            x2 = int(math.floor(x1 + n * math.cos(math.radians(angleInclination))))
            y2 = int(math.floor(y1 + n * math.sin(math.radians(angleInclination))))
    elif direction == 'B':
        # Sign Y (correct) X (may change)
        if  0 < angleInclination <= 90:
            x2 = int(math.floor(x1 - n * math.cos(math.radians(angleInclination))))
            y2 = int(math.floor(y1 + n * math.sin(math.radians(angleInclination))))
        elif  90 < angleInclination <= 180:
            x2 = int(math.floor(x1 + n * math.cos(math.radians(angleInclination))))
            y2 = int(math.floor(y1 + n * math.sin(math.radians(angleInclination))))
    return x2, y2
