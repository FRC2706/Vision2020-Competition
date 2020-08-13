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
moving_since_last_frame = False
fps = 2
contour_found = False
check_stop_count = 0
check_stop_count_max = 1 #when it's over one frame not moving, stop
time1 = 0.0 #initializing so compiler doesn't complain
time2 = 0.0
tArray = np.zeros(200)
xArray = np.zeros(200)
last_good_leftmost = -1
know_direction = False
moving_right = False

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
videoname = '../ElevatorVideos/elevator greentape moving left.mp4'

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
    contour_found, leftmost, rightmost = findMagnet(frame)
    if contour_found == True: #or else when there's no contour it yeets
        if (know_direction == True) and (moving_right == False):
            extreme = rightmost[0]
        else:
            extreme = leftmost[0]
    else:
        extreme = -1
    print('after findMagnet')
    print('leftmost = ', leftmost, ' rightmost = ', rightmost, ' extreme = ', extreme)


    if contour_found == True:

        if first_time == True:
            extreme_prev = extreme
            first_time = False
        print('extreme = ', extreme)
        print ('extreme_prev = ', extreme_prev)
        
        if (contour_found == True) and (abs(extreme - extreme_prev) > 2) : # moved more than 2 pixels from the x coordinate of leftmost
            if know_direction == False:
                if extreme > extreme_prev:
                    moving_right = True
                    print('moving_right =', moving_right)
                else:
                    moving_right = False
                    print('moving_right =', moving_right)
                know_direction = True
                if moving_right == False:
                    extreme = rightmost[0]
                print("******** direction known moving_right = ", moving_right)
            moving_since_last_frame = True
        else:
            moving_since_last_frame = False

        extreme_prev = extreme

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
            tArray[0] = time.time() #get current time - 0 position of tArray is when just started moving
            xArray[0] = extreme

        else:
            # Is continuing to move
            num_frames_moving += 1
            print('is continuing to move')
            tArray[num_frames_moving - 1] = time.time() #fill up array
            xArray[num_frames_moving - 1] = extreme

    else:
        if in_moving_state == True:
            # may have stopped
            num_frames_moving += 1
            print("num_frames_moving = ", num_frames_moving)
            tArray[num_frames_moving - 1] = time.time() 
            xArray[num_frames_moving - 1] = extreme
            if contour_found == True:    
                last_good_extreme = extreme
                print("last_good_extreme = ", last_good_extreme)
            
            check_stop_count += 1
            if check_stop_count > check_stop_count_max:
                #we're really stopped
                if moving_right == True:
                    three_quarter_location = xArray[0] + 0.75*abs(last_good_extreme - xArray[0]) #finding doors 3/4 open - abs compensate for both directions
                else:
                    three_quarter_location = xArray[0] - 0.75*abs(last_good_extreme - xArray[0])
                print("three quarter location(pixel) = ", three_quarter_location)
                min_dist = math.inf;
                for i in range(len(xArray)):
                    dist = abs(three_quarter_location - xArray[i])
                    if (dist <= min_dist): #have to do <= because the door will not move for 2 frames - then it'll be a remainder of 0 if subtract
                        min_dist = dist
                    else:
                        index_three_quarters = i - 1  # subtract 1 since we have gone 1 past minimum distance
                        break;
                time_three_quarters = tArray[index_three_quarters] - tArray[0]
                print("tArray[index_three_quarters] = ", tArray[index_three_quarters])
                print("index_three_quarters = ", index_three_quarters)
                print("time_three_quarters = ", time_three_quarters)
                print("tArray = ", tArray)
                print("xArray = ", xArray)
                #total_time_moving = (num_frames_moving - (check_stop_count + 1)) * (1.0/fps)
                #print("total_time_moving = ", total_time_moving)
                in_moving_state = False
                print('stopped moving')
                k = cv2.waitKey(0)
        # else nothing to update 

    print("know direction = ", know_direction, " moving_right = ", moving_right)
    print("xArray[0:10]=", xArray[0:10])

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
