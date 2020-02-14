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
from threading import Thread
import random

import cv2
import numpy as np

import math

import os

# Imports EVERYTHING from these files
from FindBall import *
from FindTarget import *
from VisionConstants import *
from VisionUtilities import *
from VisionMasking import *
from DistanceFunctions import *
from ControlPanel import *

###################### PROCESSING OPENCV ################################

# counts frames for writing images
frameStop = 0
ImageCounter = 0

#Code to load images from a folder
def load_images_from_folder(folder):
    images = []
    imagename = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            imagename.append(filename)
    return images, imagename

# Power Cell Images
#images, imagename = load_images_from_folder("./PowerCell25Scale")
#images, imagename = load_images_from_folder("./PowerCellImages")
#images, imagename = load_images_from_folder("./PowerCellFullScale")
#images, imagename = load_images_from_folder("./PowerCellFullMystery")
#images, imagename = load_images_from_folder("./PowerCellSketchup")
#images, imagename = load_images_from_folder("./LifeCamPhotos")

# Outer Target Images
#images, imagename = load_images_from_folder("./OuterTargetImages")
#images, imagename = load_images_from_folder("./OuterTargetHalfScale")
#images, imagename = load_images_from_folder("./OuterTargetRingTest")
images, imagename = load_images_from_folder("./OuterTargetFullDistance")
#images, imagename = load_images_from_folder("./OuterTargetHalfDistance")
#images, imagename = load_images_from_folder("./OuterTargetSketchup")
#images, imagename = load_images_from_folder("./OuterTargetLiger")

# finds height/width of camera frame (eg. 640 width, 480 height)
image_height, image_width = images[0].shape[:2]
print(image_height, image_width)

team = 2706
server = False
cameraConfigs = []

currentImg = 0

Driver = False
Tape = True
PowerCell = False
ControlPanel = False

img = images[0]
filename = imagename[0]

imgLength = len(images)

print("Hello Vision Team!")

while True:
    frame = img
    if Driver:
        processed = frame
    else:
        if Tape:
            threshold = threshold_video(lower_green, upper_green, frame)
            processed = findTargets(frame, threshold)

        else:
            if PowerCell:
                boxBlur = blurImg(frame, yellow_blur)
                threshold = threshold_video(lower_yellow, upper_yellow, boxBlur)
                processed = findPowerCell(frame, threshold)
            elif ControlPanel:
                boxBlur = blurImg(frame, yellow_blur)
                threshold = threshold_video(lower_yellow, upper_yellow, frame)
                processed = findControlPanel(frame, threshold)

    cv2.imshow("raw", img)
    cv2.imshow(filename, processed)
    cv2.setMouseCallback('raw', draw_circle)

    key = cv2.waitKey(0)
    print(key) 

    if key == 27:
        break

    currentImg += 1
    print(imgLength)

    if (currentImg == imgLength):
         currentImg = 0

    img = images[currentImg]

    #destroy old window
    cv2.destroyWindow(filename)
    filename = imagename[currentImg]
    