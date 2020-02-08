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

import json
import time
import sys
from threading import Thread
import random


import cv2
import numpy as np

import math

import os

# from https://www.pyimagesearch.com/2015/08/10/checking-your-opencv-version-using-python/
def is_cv3():
    # if we are using OpenCV 3.X, then our cv2.__version__ will start
    # with '3.'
    return check_opencv_version("3.")
 
def is_cv4():
    # if we are using OpenCV 4.X, then our cv2.__version__ will start
    # with '4.'
    return check_opencv_version("4.")

def check_opencv_version(major, lib=None):
    # if the supplied library is None, import OpenCV
    if lib is None:
        import cv2 as lib
    # return whether or not the current OpenCV version matches the
    # major version number
    return lib.__version__.startswith(major) 


########### SET RESOLUTION TO 640x480 !!!! ############

# import the necessary packages
import datetime


# Class to examine Frames per second of camera stream. Currently not used.


###################### PROCESSING OPENCV ################################

# counts frames for writing images
frameStop = 0
ImageCounter = 0

# Angles in radians

# image size ratioed to 4:3


# Lifecam 3000 from datasheet
# Datasheet: https://dl2jx7zfbtwvr.cloudfront.net/specsheets/WEBC1010.pdf

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
images, imagename = load_images_from_folder("./OuterTargetHalfScale")
#images, imagename = load_images_from_folder("./OuterTargetHalfDistance")
#images, imagename = load_images_from_folder("./OuterTargetSketchup")
#images, imagename = load_images_from_folder("./OuterTargetLiger")


# finds height/width of camera frame (eg. 640 width, 480 height)
image_height, image_width = images[0].shape[:2]
print(image_height, image_width)

# FOV of microsoft camera (68.5 is camera spec)
diagonalView = math.radians(68.5)

print("Diagonal View:" + str(diagonalView))

# 4:3 aspect ratio
horizontalAspect = 4
verticalAspect = 3

# Reasons for using diagonal aspect is to calculate horizontal field of view.
diagonalAspect = math.hypot(horizontalAspect, verticalAspect)
# Calculations: http://vrguy.blogspot.com/2013/04/converting-diagonal-field-of-view-and.html
horizontalView = math.atan(math.tan(diagonalView / 2) * (horizontalAspect / diagonalAspect)) * 2
verticalView = math.atan(math.tan(diagonalView / 2) * (verticalAspect / diagonalAspect)) * 2

# Focal Length calculations: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_165
H_FOCAL_LENGTH = image_width / (2 * math.tan((horizontalView / 2)))
V_FOCAL_LENGTH = image_height / (2 * math.tan((verticalView / 2)))
# blurs have to be odd
green_blur = 1
orange_blur = 27
yellow_blur = 1

# define range of green of retroreflective tape in HSV
lower_green = np.array([55, 55, 33])
upper_green = np.array([100, 255, 255])

# define range of green of retroreflective tape in HSV
#lower_green = np.array([23, 50, 35])
#upper_green = np.array([85, 255, 255])

lower_yellow = np.array([14, 150, 100])
upper_yellow = np.array([30, 255, 255])

# real world dimensions of the goal target
# These are the full dimensions around both strips
TARGET_STRIP_LENGTH = 19.625    # inches
TARGET_HEIGHT = 17.0            # inches@!
TARGET_TOP_WIDTH = 39.25        # inches
TARGET_BOTTOM_WIDTH = TARGET_TOP_WIDTH - 2*TARGET_STRIP_LENGTH*math.cos(math.radians(60))

#This is the X position difference between the upper target length and corner point
TARGET_BOTTOM_CORNER_WIDTH = math.sqrt(math.pow(TARGET_STRIP_LENGTH,2) - math.pow(TARGET_HEIGHT,2))

# [0, 0] is center of the quadrilateral drawn around the high goal target
# [top_left, bottom_left, bottom_right, top_right]
# real_world_coordinates = np.array([
#     [-TARGET_TOP_WIDTH / 2, TARGET_HEIGHT / 2, 0.0],
#     [-TARGET_BOTTOM_WIDTH / 2, -TARGET_HEIGHT / 2, 0.0],
#     [TARGET_BOTTOM_WIDTH / 2, -TARGET_HEIGHT / 2, 0.0],
#     [TARGET_TOP_WIDTH / 2, TARGET_HEIGHT / 2, 0.0]
# ])


# real_world_coordinates = np.array([
#     [-TARGET_TOP_WIDTH / 2, TARGET_HEIGHT / 2, 0.0],
#     [-TARGET_BOTTOM_WIDTH / 2, -TARGET_HEIGHT / 2, 0.0],
#     [TARGET_BOTTOM_WIDTH / 2, -TARGET_HEIGHT / 2, 0.0],
#     [TARGET_TOP_WIDTH / 2, TARGET_HEIGHT / 2, 0.0]
# ])

#top_left, top_right, bottom_left, bottom_right
real_world_coordinates = np.array([
        [0.0, 0.0, 0.0],             # Top Left point
        [TARGET_TOP_WIDTH, 0.0, 0.0],           # Top Right Point
        [TARGET_BOTTOM_CORNER_WIDTH, TARGET_HEIGHT, 0.0],            # Bottom Left point
        [TARGET_TOP_WIDTH-TARGET_BOTTOM_CORNER_WIDTH, TARGET_HEIGHT, 0.0]          # Bottom Right point
    ])

real_world_coordinates_left = np.array([
        [0.0, 0.0, 0.0],             # Top Left point
        [TARGET_TOP_WIDTH, 0.0, 0.0],           # Top Right Point
        [TARGET_BOTTOM_CORNER_WIDTH, TARGET_HEIGHT, 0.0],         # Bottom Left point
        [TARGET_BOTTOM_CORNER_WIDTH, TARGET_HEIGHT, 0.0]          # Bottom Left point
    ])

real_world_coordinates_right = np.array([
        [0.0, 0.0, 0.0],             # Top Left point
        [TARGET_TOP_WIDTH, 0.0, 0.0],           # Top Right Point
        [TARGET_TOP_WIDTH-TARGET_BOTTOM_CORNER_WIDTH, TARGET_HEIGHT, 0.0],    # Bottom Left point
        [TARGET_TOP_WIDTH-TARGET_BOTTOM_CORNER_WIDTH, TARGET_HEIGHT, 0.0]     # Bottom Right point
    ])    




# Flip image if camera mounted upside down
def flipImage(frame):
    return cv2.flip(frame, -1)


# Blurs frame
def blurImg(frame, blur_radius):
    img = frame.copy()
    blur = cv2.blur(img, (blur_radius, blur_radius))
    return blur


def threshold_range(im, lo, hi):
    unused, t1 = cv2.threshold(im, lo, 255, type=cv2.THRESH_BINARY)
    unused, t2 = cv2.threshold(im, hi, 255, type=cv2.THRESH_BINARY_INV)
    return cv2.bitwise_and(t1, t2)


# Masks the video based on a range of hsv colors
# Takes in a frame, range of color, and a blurred frame, returns a masked frame
def threshold_video(lower_color, upper_color, blur):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    h = threshold_range(h, lower_color[0], upper_color[0])
    s = threshold_range(s, lower_color[1], upper_color[1])
    v = threshold_range(v, lower_color[2], upper_color[2])
    combined_mask = cv2.bitwise_and(h, cv2.bitwise_and(s, v))
    
    #show the mask
    cv2.imshow("mask", combined_mask)

    # hold the HSV image to get only red colors
    # mask = cv2.inRange(combined, lower_color, upper_color)

    # Returns the masked imageBlurs video to smooth out image

    return combined_mask


# Finds the tape targets from the masked image and displays them on original stream + network tales
def findTargets(frame, mask):

    # Finds contours
    if is_cv3():
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    # Take each frame
    # Gets the shape of video
    screenHeight, screenWidth, _ = frame.shape
    # Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    # Copies frame and stores it in image
    image = frame.copy()
    # Processes the contours, takes in (contours, output_image, (centerOfImage)
    if len(contours) != 0:
        image = findTape(contours, image, centerX, centerY)
    # Shows the contours overlayed on the original video
    return image


# Finds the balls from the masked image and displays them on original stream + network tables
def findPowerCell(frame, mask):
    # Finds contours
    if is_cv3():
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

    # Take each frame
    # Gets the shape of video
    screenHeight, screenWidth, _ = frame.shape
    # Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    # Copies frame and stores it in image
    image = frame.copy()
    # Processes the contours, takes in (contours, output_image, (centerOfImage)
    if len(contours) != 0:
        image = findBall(contours, image, centerX, centerY)
    # Shows the contours overlayed on the original video
    return image



# Draws Contours and finds center and yaw of orange ball
# centerX is center x coordinate of image
# centerY is center y coordinate of image
def findBall(contours, image, centerX, centerY):
    screenHeight, screenWidth, channels = image.shape
    # Seen vision targets (correct angle, adjacent to each other)
    #cargo = []

    if len(contours) > 0:
        # Sort contours by area size (biggest to smallest)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:5]
        cntHeight = 0
        biggestPowerCell = []
        for cnt in cntsSorted:
            x, y, w, h = cv2.boundingRect(cnt)

            #print("bounding rec height: " + str(h))
            #print("bounding rec width: " + str(w))
        
            cntHeight = h
            aspect_ratio = float(w) / h
            # Get moments of contour; mainly for centroid
            M = cv2.moments(cnt)
            # Get convex hull (bounding polygon on contour)
            #hull = cv2.convexHull(cnt)
            # Calculate Contour area
            cntArea = cv2.contourArea(cnt)
            # Filters contours based off of size
            if (checkBall(cntArea, aspect_ratio)):
                ### MOSTLY DRAWING CODE, BUT CALCULATES IMPORTANT INFO ###
                # Gets the centeroids of contour
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                if (len(biggestPowerCell) < 3):

                    ##### DRAWS CONTOUR######
                    # Gets rotated bounding rectangle of contour
                    rect = cv2.minAreaRect(cnt)
                    # Creates box around that rectangle
                    box = cv2.boxPoints(rect)
                    # Covert boxpoints to integer
                    box = np.int0(box)
                    # Draws rotated rectangle
                    #cv2.drawContours(image, [box], 0, (23, 184, 80), 3)

                    # Draws a vertical white line passing through center of contour
                    cv2.line(image, (cx, screenHeight), (cx, 0), (255, 255, 255))
                    # Draws a white circle at center of contour
                    cv2.circle(image, (cx, cy), 6, (255, 255, 255))

                    # Draws the contours
                    cv2.drawContours(image, [cnt], 0, (23, 184, 80), 1)

                    # Draws contour of bounding rectangle in red
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                   
                    # Appends important info to array
                    if [cx, cy, cnt, cntHeight] not in biggestPowerCell:
                        biggestPowerCell.append([cx, cy, cnt, cntHeight, aspect_ratio])

        # Check if there are PowerCell seen
        if (len(biggestPowerCell) > 0):
            # pushes that it sees cargo to network tables

            finalTarget = []
            # Sorts targets based on largest height
            biggestPowerCell.sort(key=lambda height: math.fabs(height[3]))

            #sorts closestPowerCell - contains center-x, center-y, contour and contour height from the
            #bounding rectangle.  The closest one has the largest height
            closestPowerCell = min(biggestPowerCell, key=lambda height: (math.fabs(height[3] - centerX)))

            # extreme points
            leftmost = tuple(closestPowerCell[2][closestPowerCell[2][:,:,0].argmin()][0])
            rightmost = tuple(closestPowerCell[2][closestPowerCell[2][:,:,0].argmax()][0])
            topmost = tuple(closestPowerCell[2][closestPowerCell[2][:,:,1].argmin()][0])
            bottommost = tuple(closestPowerCell[2][closestPowerCell[2][:,:,1].argmax()][0])

            # draw extreme points
            # from https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
            cv2.circle(image, leftmost, 6, (0,255,0), -1)
            cv2.circle(image, rightmost, 6, (0,0,255), -1)
            cv2.circle(image, topmost, 6, (255,255,255), -1)
            cv2.circle(image, bottommost, 6, (255,0,0), -1)
            #print('extreme points', leftmost,rightmost,topmost,bottommost)

            print("topmost: " + str(topmost[0]))
            print("bottommost: " + str(bottommost[0]))
           
            #xCoord of the closest ball will be the x position differences between the topmost and 
            #bottom most points
            if (topmost[0] > bottommost[0]):
                xCoord = int(round((topmost[0]-bottommost[0])/2)+bottommost[0])
            else: 
                xCoord = int(round((bottommost[0]-topmost[0])/2)+topmost[0])

            print(xCoord)
            if (closestPowerCell[4] > 0.9 and closestPowerCell[4] < 1.2):
                xCoord = closestPowerCell[0]

            print ("aspect ratio of ball: " + str(closestPowerCell[4]))     

            finalTarget.append(calculateYaw(xCoord, centerX, H_FOCAL_LENGTH))
            finalTarget.append(calculateDistWPILib(closestPowerCell[3]))
            #print("Yaw: " + str(finalTarget[0]))

            # Puts the yaw on screen
            # Draws yaw of target + line where center of target is
            finalYaw = round(finalTarget[1]*1000)/1000
            cv2.putText(image, "Yaw: " + str(finalTarget[0]), (40, 40), cv2.FONT_HERSHEY_COMPLEX, .6,
                        (255, 255, 255))
            cv2.putText(image, "Dist: " + str(finalYaw), (40, 100), cv2.FONT_HERSHEY_COMPLEX, .6,
                        (255, 255, 255))
            cv2.line(image, (xCoord, screenHeight), (xCoord, 0), (255, 0, 0), 2)

            currentAngleError = finalTarget[0]
            # pushes cargo angle to network tables



        cv2.line(image, (round(centerX), screenHeight), (round(centerX), 0), (255, 255, 255), 2)

        return image


def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
 
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[3] = pts[np.argmax(s)]
 
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[2] = pts[np.argmax(diff)]
 
    # return the ordered coordinates
    return rect

 #3D Rotation estimation
def findTvecRvec(image, outer_corners, real_world_coordinates):
    # Read Image
    size = image.shape
 
    # Camera internals
 
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    # camera_matrix = np.array(
    #                      [[H_FOCAL_LENGTH, 0, center[0]],
    #                      [0, V_FOCAL_LENGTH, center[1]],
    #                      [0, 0, 1]], dtype = "double"
    #                      )


    dist_coeffs = np.array([[0.16171335604097975, -0.9962921370737408, -4.145368586842373e-05, 
                             0.0015152030328047668, 1.230483016701437]])

    camera_matrix = np.array([[676.9254672222575, 0.0, 303.8922263320326], 
                              [0.0, 677.958895098853, 226.64055316186037], 
                              [0.0, 0.0, 1.0]], dtype = "double")

    print("Camera Matrix :\n {0}".format(camera_matrix))                           
 
    #dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(real_world_coordinates, outer_corners, camera_matrix, dist_coeffs)
 
    #print ("Rotation Vector:\n {0}".format(rotation_vector))
    #print ("Translation Vector:\n {0}".format(translation_vector))
    return success, rotation_vector, translation_vector


#Computer the final output values, 
#angle 1 is the Yaw to the target
#distance is the distance to the target
#angle 2 is the Yaw of the Robot to the target
def compute_output_values(rvec, tvec):
    '''Compute the necessary output distance and angles'''

    # The tilt angle only affects the distance and angle1 calcs
    # This is a major impact on calculations
    tilt_angle = math.radians(35)

    x = tvec[0][0]
    z = math.sin(tilt_angle) * tvec[1][0] + math.cos(tilt_angle) * tvec[2][0]

    # distance in the horizontal plane between camera and target
    distance = math.sqrt(x**2 + z**2)

    # horizontal angle between camera center line and target
    angleInRad = math.atan2(x, z)
    angle1 = math.degrees(angleInRad)

    rot, _ = cv2.Rodrigues(rvec)
    rot_inv = rot.transpose()
    pzero_world = np.matmul(rot_inv, -tvec)
    angle2 = math.atan2(pzero_world[0][0], pzero_world[2][0])

    return distance, angle1, angle2


# Draws Contours and finds center and yaw of vision targets
# centerX is center x coordinate of image
# centerY is center y coordinate of image
# Draws Contours and finds center and yaw of vision targets
# centerX is center x coordinate of image
# centerY is center y coordinate of image
def findTape(contours, image, centerX, centerY):

    #global warped
    screenHeight, screenWidth, channels = image.shape
    # Seen vision targets (correct angle, adjacent to each other)
    targets = []

    if len(contours) >= 1:
        # Sort contours by area size (biggest to smallest)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:1]
       
        for cnt in cntsSorted:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

            # rotated rectangle
            rect = cv2.minAreaRect(cnt)
            #print('rotated rectangle = ',rect)
            (x,y),(width,height),angleofrotation = rect
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(image,[box],0,(255,0,0),2)
            #print("box points: " + str(box))

            hull = cv2.convexHull(cnt)
            #print('hull', hull)
            print('hull contour length = ', len(hull))
            #cv2.drawContours(image, [hull], -1, (0,0,255), cv2.FILLED)
            #cv2.imshow('hull over yellow mask', imgContours)
            hull_area = cv2.contourArea(hull)
            #print('solidity from convex hull', float(area)/hull_area)

            # extreme points
            leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
            rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
            topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
            bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

            bottomIsLeft = True
            #check if bottommost is closest to right or left
            if (abs(bottommost[0]-leftmost[0]) > abs(bottommost[0]-rightmost[0])):
                print("bottom most is right")
                bottomIsLeft = False

            else:
                 print("bottom most is left")

            # draw extreme points
            # from https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
            cv2.circle(image, leftmost, 6, (0,255,0), -1)
            cv2.circle(image, rightmost, 6, (0,0,255), -1)
            cv2.circle(image, topmost, 6, (255,255,255), -1)
            cv2.circle(image, bottommost, 6, (255,0,0), -1)
            #print('extreme points', leftmost,rightmost,topmost,bottommost)

    
            #Set up the 3 points to map to the real world coordinates
            outer_corners = np.array([leftmost, rightmost, bottommost, bottommost], dtype="double")
            print("points: " + str(outer_corners))

           #sorted_corners = order_points(outer_corners)
            #print("sorted corners: " + str(sorted_corners))

            if (bottomIsLeft):
                success, rvec, tvec = findTvecRvec(image, outer_corners, real_world_coordinates_left) 
            else:
                success, rvec, tvec = findTvecRvec(image, outer_corners, real_world_coordinates_right) 

            #Calculate the Yaw
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = 0, 0

            YawToTarget = calculateYaw(cx, centerX, H_FOCAL_LENGTH)
            
            # If success then print values to screen                               
            if success:
                distance, angle1, angle2 = compute_output_values(rvec, tvec)
                cv2.putText(image, "TargetYawToCenter: " + str(YawToTarget), (40, 340), cv2.FONT_HERSHEY_COMPLEX, .6,(255, 255, 255))
                cv2.putText(image, "Distance: " + str(distance/12), (40, 380), cv2.FONT_HERSHEY_COMPLEX, .6,(255, 255, 255))
                #cv2.putText(image, "RobotYawToTarget: " + str(angle2), (40, 420), cv2.FONT_HERSHEY_COMPLEX, .6,(255, 255, 255))
                cv2.line(image, (cx, screenHeight), (cx, 0), (255, 0, 0), 2)
                cv2.line(image, (round(centerX), screenHeight), (round(centerX), 0), (255, 255, 255), 2)

    #     # pushes vision target angle to network table
    return image


# Finds the balls from the masked image and displays them on original stream + network tables
def findControlPanel(frame, mask):
    # Finds contours
    if is_cv3:
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    # Take each frame
    # Gets the shape of video
    screenHeight, screenWidth, _ = frame.shape
    # Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    # Copies frame and stores it in image
    image = frame.copy()
    # Processes the contours, takes in (contours, output_image, (centerOfImage)
    if len(contours) != 0:
        image = findControlPanelColour(contours, image, centerX, centerY)
    # Shows the contours overlayed on the original video
    return image

# Draws Contours and finds the colour the control panel wheel is resting at
# centerX is center x coordinate of image
# centerY is center y coordinate of image
def findControlPanelColour(contours, image, centerX, centerY):
    #ToDo, Add code to publish wheel colour
    return image

# Checks if tape contours are worthy based off of contour area and (not currently) hull area
def checkContours(cntSize, hullSize):
    print(cntSize, image_width / 7)
    return cntSize > (image_width / 7)


# Checks if ball contours are worthy based off of contour area and (not currently) hull area
def checkBall(cntSize, cntAspectRatio):
    #this checks that the area of the contour is greater than the image width divide by 2
    #And that the aspect ratio of the bounding rectangle (width / height) is close to 1 which 
    #is basically a circle however this would filter out 'tadpoles'
    
   # return (cntSize > (image_width / 2)) and (round(cntAspectRatio) > 1)
    return (cntSize > (image_width / 2)) and (cntAspectRatio > 0.75)


# Forgot how exactly it works, but it works!
def translateRotation(rotation, width, height):
    if (width < height):
        rotation = -1 * (rotation - 90)
    if (rotation > 90):
        rotation = -1 * (rotation - 180)
    rotation *= -1
    return round(rotation)


def calculateDistance(heightOfCamera, heightOfTarget, pitch):
    heightOfTargetFromCamera = heightOfTarget - heightOfCamera

    # Uses trig and pitch to find distance to target
    '''
    d = distance
    h = height between camera and target
    a = angle = pitch
    tan a = h/d (opposite over adjacent)
    d = h / tan a
                         .
                        /|
                       / |
                      /  |h
                     /a  |
              camera -----
                       d
    '''
    divisor = math.tan(math.radians(pitch))
    distance = 0
    if (divisor != 0):
        distance = math.fabs(heightOfTargetFromCamera / divisor)

    return distance


avg = [0 for i in range(0, 1)]
#8 is number of frames to calculated average pixel height

def calculateDistWPILib(cntHeight):
    global image_height, avg

    for cnt in avg:
        if cnt == 0:
            cnt = cntHeight

    del avg[len(avg) - 1]
    avg.insert(0, cntHeight)
    PIX_HEIGHT = 0
    for cnt in avg:
        PIX_HEIGHT += cnt

    PIX_HEIGHT = PIX_HEIGHT / len(avg)

    print (PIX_HEIGHT)



    print(PIX_HEIGHT, avg)  # print("The contour height is: ", cntHeight)

    #TARGET_HEIGHT is actual height (for balls 7/12 7 inches)   
    TARGET_HEIGHT = 0.583

 
    #image height is the y resolution calculated from image size
    #15.81 was the pixel height of a a ball found at a measured distance (which is 6 feet away)
    #65 is the pixel height of a scale image 6 feet away
    KNOWN_OBJECT_PIXEL_HEIGHT = 65
    KNOWN_OBJECT_DISTANCE = 6
    VIEWANGLE = math.atan((TARGET_HEIGHT * image_height) / (2 * KNOWN_OBJECT_PIXEL_HEIGHT * KNOWN_OBJECT_DISTANCE))

    # print("after 2: ", VIEWANGLE)
    # VIEWANGLE = math.radians(68.5)
    distance = ((TARGET_HEIGHT * image_height) / (2 * PIX_HEIGHT * math.tan(VIEWANGLE)))

    return distance


# Uses trig and focal length of camera to find yaw.
# Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
def calculateYaw(pixelX, centerX, hFocalLength):
    yaw = math.degrees(math.atan((pixelX - centerX) / hFocalLength))
    return round(yaw)


# Link to further explanation: https://docs.google.com/presentation/d/1ediRsI-oR3-kwawFJZ34_ZTlQS2SDBLjZasjzZ-eXbQ/pub?start=false&loop=false&slide=id.g12c083cffa_0_298
def calculatePitch(pixelY, centerY, vFocalLength):
    pitch = math.degrees(math.atan((pixelY - centerY) / vFocalLength))
    # Just stopped working have to do this:
    pitch *= -1
    return round(pitch)


def getEllipseRotation(image, cnt):
    try:
        # Gets rotated bounding ellipse of contour
        ellipse = cv2.fitEllipse(cnt)
        centerE = ellipse[0]
        # Gets rotation of ellipse; same as rotation of contour
        rotation = ellipse[2]
        # Gets width and height of rotated ellipse
        widthE = ellipse[1][0]
        heightE = ellipse[1][1]
        # Maps rotation to (-90 to 90). Makes it easier to tell direction of slant
        rotation = translateRotation(rotation, widthE, heightE)

        cv2.ellipse(image, ellipse, (23, 184, 80), 3)
        return rotation
    except:
        # Gets rotated bounding rectangle of contour
        rect = cv2.minAreaRect(cnt)
        # Creates box around that rectangle
        box = cv2.boxPoints(rect)
        # Not exactly sure
        box = np.int0(box)
        # Gets center of rotated rectangle
        center = rect[0]
        # Gets rotation of rectangle; same as rotation of contour
        rotation = rect[2]
        # Gets width and height of rotated rectangle
        width = rect[1][0]
        height = rect[1][1]
        # Maps rotation to (-90 to 90). Makes it easier to tell direction of slant
        rotation = translateRotation(rotation, width, height)
        return rotation




team = 2706
server = False
cameraConfigs = []

currentImg = 0

def draw_circle(event,x,y,flags,param):


    if event == cv2.EVENT_LBUTTONDOWN:
        green = np.uint8([[[img[y, x, 0], img[y, x, 1], img[y, x, 2]]]])
        print(img[y, x, 2], img[y, x, 1], img[y, x, 0], cv2.cvtColor(green,cv2.COLOR_BGR2HSV))


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
                # cv2.putText(frame, "Find Cargo", (40, 40), cv2.FONT_HERSHEY_COMPLEX, .6, (255, 255, 255))
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
    






