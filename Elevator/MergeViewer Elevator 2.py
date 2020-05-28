# ----------------------------------------------------------------------------
# Copyright (c) 2018 FIRST. All Rights Reserved.
# Open Source Software - may be modified and shared by FRC teams. The code
# must be accompanied by the FIRST BSD license file in the root directory of
# the project.

# My 2020 license: use it as much as you want. Crediting is recommended because it lets me know 
# that I am being useful.
# Some parts of pipeline are based on 2019 code created by the Screaming Chickens 3997

# This is meant to be used in conjuction with WPILib Raspberry Pi image: https://github.com/wpilibsuite/FRCVision-pi-gen
# ----------------------------------------------------------------------------

# import the necessary packages
import datetime
import json
import time
import sys
import random
import cv2
import math
import os
import sys

import numpy as np

print("Using python version {0}".format(sys.version))
print('OpenCV Version = ', cv2.__version__)
print()

# Imports EVERYTHING from these files
from FindMagnet import *

###################### PROCESSING OPENCV ################################

# counts frames for writing images
frameStop = 0
ImageCounter = 0
showAverageFPS = False
first_time = True
in_moving_state = False
fps = 2
found = False
check_stop_count = 0
check_stop_count_max = 1 #when it's over one frame not moving, stop

# CHOOSE VIDEO OR FILES HERE!!!!
# boolean for video input, if true does video, if false images
useVideo = True
# integer for usb camera to use, boolean for live webcam
useWebCam = False
webCamNumber = 1

#Code to load images from a folder
def load_images_from_folder(folder):
    images = []
    imagename = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            imagename.append(filename)
    return images, imagename

def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        green = np.uint8([[[img[y, x, 0], img[y, x, 1], img[y, x, 2]]]])
        print(img[y, x, 2], img[y, x, 1], img[y, x, 0], cv2.cvtColor(green,cv2.COLOR_BGR2HSV))  

# choose video to process -> Outer Target Videos
#videoname = './OuterTargetVideos/ThirdScale-01.mp4'
videoname = '../ElevatorVideos/elevator greentape trimmed.mp4'

if useVideo: # test against video
    showAverageFPS = True

elif useWebCam: #test against live camera
    showAverageFPS = True

else:  # implies images are to be read

    # This is the one with the power cell
    images, imagename = load_images_from_folder("../ControlPanelSquares")


    # finds height/width of camera frame (eg. 640 width, 480 height)
    image_height, image_width = images[0].shape[:2]
    print(image_height, image_width)

    currentImg = 0

team = 2706
server = False
cameraConfigs = []

Driver = False
Tape = False
PowerCell = True
ControlPanel = False


if useVideo and not useWebCam:
    cap = cv2.VideoCapture(videoname)

elif useWebCam:
    # src defines which camera, assume 2nd camera or src=1
    vs = WebcamVideoStream(src=webCamNumber).start()

else:
    img = images[0]
    filename = imagename[0]
    imgLength = len(images)

print("Hello Vision Team!")

stayInLoop = True

#Setup variables for average framecount
frameCount = 0
averageTotal = 0
averageFPS = 0

framePSGroups = 50
displayFPS = 3.14159265

# start

while stayInLoop or cap.isOpened():

    if useVideo and not useWebCam:
        (ret, frame) = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame, likely end of file, Exiting ...")
            stayInLoop = False
            break

    elif useWebCam:
        frame = vs.read()

    else:
        imgLength = len(images)
        frame = img


    #
    # Put elevator code here
    #

    print('------------------------')
    print('before findMagnet')
    found, leftmost = findMagnet(frame)
    print('after findMagnet')

    if found == True:

        if first_time == True:
            leftmost_prev = leftmost

            first_time = False
        print('leftmost = ', leftmost)
        print('leftmost_prev = ', leftmost_prev)
        if (leftmost[0] - leftmost_prev[0] > 2): # moved more than 2 pixels from the x coordinate of leftmost
            moving_since_last_frame = True
            print('is moving since last frame')
        else:
            moving_since_last_frame = False

        leftmost_prev = leftmost  # Setup for next time through loop

    else:
        moving_since_last_frame = False
        print('contour not found')

    print('moving since last frame is ', moving_since_last_frame)  
    print('in moving state (before) ', in_moving_state)  

    if moving_since_last_frame == True:
        check_stop_count = 0
        if in_moving_state == False:
            # Just started moving
            num_frames_moving = 1
            in_moving_state = True
            print('just started moving')
        else:
            # Is continuing to move
            num_frames_moving += 1
            print('is continuing to move')
    else:
        if in_moving_state == True:
            # stopped
            check_stop_count += 1
            if check_stop_count > check_stop_count_max:
                #we're really stopped
                total_time_moving = (num_frames_moving - (check_stop_count + 1)) * (1.0/fps)
                print("total_time_moving = ", total_time_moving)
                in_moving_state = False
                print('stopped moving')
                k = cv2.waitKey(0)
        # else nothing to update 

    print('in moving state (after) ', in_moving_state)  

    if useVideo or useWebCam:
        #cv2.imshow('videoname', processed)

        key = cv2.waitKey(1)
        if key == 27:
            break

    else:
        #cv2.imshow(filename, processed)

        key = cv2.waitKey(0)
        print(key) 

        if key == 27:
            stayInLoop = False
            break

        currentImg += 1
        print(imgLength)

        if (currentImg == imgLength):
            currentImg = 0

        img = images[currentImg]

        #destroy old window
        cv2.destroyWindow(filename)
        filename = imagename[currentImg]

    # end while
# end if

if useVideo and not useWebCam:
    cap.release()
elif useWebCam:
    vs.stop()
else:
    pass

cv2.destroyAllWindows()
