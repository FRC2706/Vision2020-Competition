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

#print("Using python version {0}".format(sys.version))
#print('OpenCV Version = ', cv2.__version__)
#print()

# Imports EVERYTHING from these files
from FindMagnet import *

###################### PROCESSING OPENCV ################################

# counts frames for writing images
frameStop = 0
ImageCounter = 0
showAverageFPS = False
avgFrames = [0 for i in range(0, 5)]
avgX = [0 for i in range(0, len(avgFrames))]
xCoordinate = []
timeStamps = []
tupTime = (0, 0)
start = True
index = 0
foundTape = False
counter = 0
startingFrame = True
initTime = True
averageX = 0
timeElapsed = 0
startX = 0
oldTime = 0
now = 0
started = False
timing = True
previousX = 0
closest = 0

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
        #print(img[y, x, 2], img[y, x, 1], img[y, x, 0], cv2.cvtColor(green,cv2.COLOR_BGR2HSV))  

# choose video to process -> Outer Target Videos
#videoname = './OuterTargetVideos/ThirdScale-01.mp4'
videoname = './ElevatorVideos/FifthScale-54CO.mp4'

if useVideo: # test against video
    showAverageFPS = True

elif useWebCam: #test against live camera
    showAverageFPS = True

else:  # implies images are to be read

    # This is the one with the power cell
    images, imagename = load_images_from_folder("../ControlPanelSquares")


    # finds height/width of camera frame (eg. 640 width, 480 height)
    image_height, image_width = images[0].shape[:2]
    #print(image_height, image_width)

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

#print("Hello Vision Team!")

stayInLoop = True

#Setup variables for average framecount
frameCount = 0
averageTotal = 0
averageFPS = 0

framePSGroups = 50
displayFPS = 3.14159265

# start
type = input("Closing or Opening? ")
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

    #print('------------------------')
    #print('before findMagnet')
    contour_found, leftmost, rightmost = findMagnet(frame)
    #print('after findMagnet')

    if contour_found == True:
        if timing:
            foundTape = True
            while start:
                del avgFrames[len(avgFrames) - 1]
                avgFrames.insert(0, leftmost)
                if(counter == len(avgFrames) - 1):
                    start = False
                if(counter == 0):
                    xCoordinate.insert(0, leftmost[0])
                    timeStamps.insert(0, 0.000)
                counter += 1

            del avgFrames[len(avgFrames) - 1]
            avgFrames.insert(0, leftmost)

            for x in range(0, len(avgX)):
                avgX[x] = avgFrames[x][0]

            averageX = sum(avgX) / len(avgX)
            if startingFrame:
                startX = averageX
                startingFrame = False

            if abs(averageX - startX) >= 0.5:
                started = True
                now = time.time()
                if initTime:
                    oldTime = now
                    initTime = False
                timeElapsed = round(now - oldTime,4)
                xCoordinate.insert(0, leftmost[0])
                timeStamps.insert(0, timeElapsed)
                if started & (previousX == averageX):
                    xCoordinate.insert(0, leftmost[0])
                    timeStamps.insert(0, timeElapsed)
                    closest = xCoordinate[0]
                    print('xCoordinate array:', xCoordinate)
                    print('timeStamps array:', timeStamps)
                    xDifference = xCoordinate[len(xCoordinate)-1] - xCoordinate[0]
                    3QDifference = xDifference*0.75
                    for i in range(len(xCoordinate)):
                        if abs(xCoordinate[i] - xDifference) < 2:
                            index = i
                            break
                    print('(Door Reached 75% Closed) Time Elapsed:', timeStamps[index], 'seconds')

                    timeElapsed = round(now - oldTime,2)
                    print("(Door Opened) Time Elapsed:", timeStamps[len(timeStamps)-1], "seconds")

                    started = False
                    foundTape = False
                    timing = False
                    if type == "closing":
                        break
            previousX = averageX

    else:
        if foundTape:
            print('xCoordinate array:', xCoordinate)
            print('timeStamps array:', timeStamps)
            xDifference = xCoordinate[len(xCoordinate)-1] - xCoordinate[0]
            for i in range(len(xCoordinate)):
                if abs(xCoordinate[i] - xDifference) < 2:
                    index = i
                    break
            print('(Door Reached 75% Opened) Time Elapsed:', timeStamps[index], 'seconds')

            print("(Door Opened) Time Elapsed:", timeStamps[len(timeStamps)-1], "seconds")
            foundTape = False
            if type == "opening":
                break
            

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