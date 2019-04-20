'''
Created on May 26, 2016

@author: pankajrawat
'''
import os

DEBUG = False
DISPLAY_COLOR = False
TRAIN_MODE = False

ESTIMATER_PICKLE_FILE = os.path.join('classifiers', 'SVC')
TRAIN_OUTPUT_FILE = os.path.join("dataset", "data.csv")
OUTPUT_IMAGE_PATH = os.path.join("sampleImages", "result.jpg")


""" Detecting Line """
REPAIR_MIN_AREA_REQUIRED = 15
REPAIR_FLAG_DISPLAY_LINE_MIDPOINS = False 
REPAIR_PIXEL_JUMP_WHILE_DETECTING_LINE = 6  #Number of pixels to jump when detecting whether contour is line
REPAIR_MAX_PIXEL_POINT_JUMP = 25   # Max number of jumps
REPAIR_MAX_ANGLE_DIFFERENCE_ALLOWED = 20
REPAIR_MAX_LINE_ERROR = 0.2
REPAIR_MIN_SEGMENT_PASS = 5
REPAIR_IMMEDIATE_PIXEL_SKIP = 3
REPAIR_MIN_OTHER_COUNTER_POINT_REQUIRED_FOR_JOINING = 3


""" Detecting blobs """
BREAK_MIN_AREA_REQUIRED = 60
BREAK_PERCENTILE_BLOB = 97
BREAK_PERCENTILE_THRESHOLD = 0.75

BLACK = [0, 0, 0]
BLUE = [255, 0, 0]
RED = [0, 0, 255]
GREEN = [0, 255, 0]
YELLOW = [0, 255, 255]
ORANGE = [0, 120, 255]
GREY = [128, 128, 128]
WHITE = [255, 255, 255]

IMAGE_DIMENTION = 128
TRAINING_AREA_THRESHOLD = 10
