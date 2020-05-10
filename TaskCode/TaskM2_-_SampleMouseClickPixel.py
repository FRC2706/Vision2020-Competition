# This is a pseudo code file for Merge Robotics, 2020, Infinite Recharge
# This is task M2 - > Sample Mouse Clicks by Pixel. 
# We are going to continue towards our objective of a tool for season kickoff
# This pseudo file will allow us to determine a pixel color by clicking
# on it.  The purpose is to help calibration of the color filter.
# Using web searches for python and pixel color with a mouse, create your own
# code to deliver this capacity.

# online info we already know!
# https://docs.opencv.org/3.4.7/d3/df2/tutorial_py_basic_ops.html
# https://docs.opencv.org/3.4.7/db/d5b/tutorial_py_mouse_handling.html

# online definition of numpy addressing
# https://numpy.org/devdocs/user/quickstart.html

# copy in task M, change it for 2706-Elimins-Images

# imports
import numpy as np
import cv2
from pathlib import Path
import sys
import os

# definitions of ...
# from Merge ChickenVision 2019
def threshold_range(im, lo, hi):
    unused, t1 = cv2.threshold(im, lo, 255, type=cv2.THRESH_BINARY)
    unused, t2 = cv2.threshold(im, hi, 255, type=cv2.THRESH_BINARY_INV)
    return cv2.bitwise_and(t1, t2)

# add callback function for images we wish to query
# BGR pixel mouse callback function on imgImageInput
def onmouse_bgr_pixel_color(event,x,y,flags,params):
    global imgDoubleInput
    if event == cv2.EVENT_LBUTTONDOWN:
        print('coords',(x,y),'BGR =',imgDoubleInput[y,x])

# HSV pixel mouse callback function on hsvImageInput
def onmouse_hsv_pixel_color(event,x,y,flags,params):
    global imgDoubleHSV
    if event == cv2.EVENT_LBUTTONDOWN:
        print('coords',(x,y),'HSV =',imgDoubleHSV[y,x])

# mask pixel mouse callback function on yellow_mask
def onmouse_mas_pixel_color(event,x,y,flags,params):
    global yellow_mask
    if event == cv2.EVENT_LBUTTONDOWN:
        print('coords',(x,y),'HSV =',yellow_mask[y,x])

# select folder of interest
posCodePath = Path(__file__).absolute()
strVisionRoot = posCodePath.parent
strImageFolder = str(strVisionRoot / 'ControlPanelSquares' / 'panel-02.png')
print (strImageFolder)

# read file names, and filter file names
photos = []
if os.path.exists(strImageFolder):
    for file in sorted(os.listdir(strImageFolder)):
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".JPG") or file.endswith(".PNG"):
            photos.append(file)
else:
    print
    print ('Directory', strImageFolder, 'does not exist, exiting ...')
    print
    sys.exit
print (photos)

# set index of files
i = 0
intLastFile = len(photos) -1

# begin main loop indent 1
while (True):

    ## set image input to indexed list
    strImageInput = strImageFolder + '/' + photos[i]
    print (i, ' ', strImageInput)

    ## store filename for window title
    strDoubleImageName = photos[i]

    ## read file
    imgImageInput = cv2.imread(strImageInput)

    ## name the window in advance and create callback
    cv2.namedWindow(strDoubleImageName)
    cv2.setMouseCallback(strDoubleImageName, onmouse_bgr_pixel_color)
    cv2.moveWindow(strDoubleImageName,100,50)

    ## display double size original image
    imgDoubleInput = cv2.resize(imgImageInput, None, fx=1.0, fy=1.0, interpolation = cv2.INTER_AREA)
    cv2.imshow(strDoubleImageName, imgDoubleInput)

    ## Convert BGR to HSV
    hsvImageInput = cv2.cvtColor(imgImageInput, cv2.COLOR_BGR2HSV)

    ## define range of yellow color in HSV
    lower_yellow = np.array([15,150,150]) #28,150,150
    upper_yellow = np.array([23,255,255]) #32,255,255

    ## Threshold the HSV image to get only yellow colors
    binary_mask = cv2.inRange(hsvImageInput, lower_yellow, upper_yellow)

    ## mask the image to only show yellow or green images
    ## Bitwise-AND mask and original image
    yellow_mask = cv2.bitwise_and(hsvImageInput, hsvImageInput, mask=binary_mask)

    ## name the window in advance and create callback
    cv2.namedWindow('hsvImageInput')
    cv2.setMouseCallback('hsvImageInput',onmouse_hsv_pixel_color)
    cv2.moveWindow('hsvImageInput',450,350)

    cv2.namedWindow('yellow_masked')    
    cv2.setMouseCallback('yellow_masked',onmouse_mas_pixel_color)
    cv2.moveWindow('yellow_masked',261,350)

    ## display double size original image
    imgDoubleHSV = cv2.resize(hsvImageInput, None, fx=1.0, fy=1.0, interpolation = cv2.INTER_AREA)
    cv2.imshow('hsvImageInput', imgDoubleHSV)

    ## display the masked images to screen
    cv2.imshow('hsvImageInput', imgDoubleHSV)
    cv2.imshow('binary_mask', binary_mask)
    cv2.moveWindow('binary_mask',450,170)
    cv2.imshow('yellow_masked',yellow_mask)

    ## loop for user input to close - loop indent 2 
    booReqToExit = False # true when user wants to exit
    while (True):

        ### wait for user to press key
        k = cv2.waitKey(0)
        if k == 27:
            booReqToExit = True # user wants to exit
            break
        elif k == 82: # user wants to move down list
            if i - 1 < 0:
                i = intLastFile
            else:
                i = i - 1
            break
        elif k == 84: # user wants to move up list
            if i + 1 > intLastFile:
                i = 0
            else:
                i = i + 1
            break
        elif k == 32:
            print()
            print('...repeat...')
            break
        else:
            print (k)

        ### end of loop indent 2

    ## test for exit main loop request from user
    if booReqToExit:
        break

    ## not exiting, close window before loading next photo
    cv2.destroyWindow(strDoubleImageName)
    cv2.destroyWindow('hsvImageInput')
    cv2.destroyWindow('binary_mask')
    cv2.destroyWindow('yellow_masked')

    ## end of main loop indent 1

# cleanup and exit
cv2.destroyAllWindows()
