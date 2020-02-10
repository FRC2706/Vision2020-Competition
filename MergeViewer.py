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
from scipy import stats
from scipy.interpolate import UnivariateSpline

import cv2
import numpy as np

import math

import os

########### SET RESOLUTION TO 640x480 !!!! ############

# import the necessary packages
import datetime

print("Using python version {0}".format(sys.version))
print('OpenCV Version = ', cv2.__version__)
print()

# define colors
purple = (165, 0, 120)
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
cyan = (252, 252, 3)
white = (255, 255, 255)
magenta = (255, 0, 255)

# real world dimensions of the goal target
# These are the full dimensions around both strips
TARGET_STRIP_LENGTH = 19.625    # inches
TARGET_HEIGHT = 17.0            # inches@!
TARGET_TOP_WIDTH = 39.25        # inches
TARGET_BOTTOM_WIDTH = TARGET_TOP_WIDTH - 2*TARGET_STRIP_LENGTH*math.cos(math.radians(60))

#This is the X position difference between the upper target length and corner point
TARGET_BOTTOM_CORNER_WIDTH = TARGET_STRIP_LENGTH*math.cos(math.radians(60))

real_world_coordinates = np.array([
        [0.0, 0.0, 0.0],             # Top Left point
        [TARGET_TOP_WIDTH, 0.0, 0.0],           # Top Right Point
        [TARGET_BOTTOM_CORNER_WIDTH, TARGET_HEIGHT, 0.0],            # Bottom Left point
        [TARGET_TOP_WIDTH-TARGET_BOTTOM_CORNER_WIDTH, TARGET_HEIGHT, 0.0]          # Bottom Right point
    ])

# counts frames for writing images
frameStop = 0
ImageCounter = 0

# Angles in radians

# image size ratioed to 4:3


# Lifecam 3000 from datasheet
# Datasheet: https://dl2jx7zfbtwvr.cloudfront.net/specsheets/WEBC1010.pdf

def load_images_from_folder(folder):
    images = []
    filenames = os.listdir(folder)
    for filename in filenames:
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images, filenames

#images, filenames = load_images_from_folder("./OuterTargetImages")
images, filenames = load_images_from_folder("./OuterTargetHalfScale")
#images = load_images_from_folder("./PowerCell25Scale")
#images = load_images_from_folder("./PowerCellImages")
#images = load_images_from_folder("./PowerCellFullScale")
#images = load_images_from_folder("./PowerCellFullMystery")
#images = load_images_from_folder("./PowerCellSketchup")
#images = load_images_from_folder("./LifeCamPhotos")

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
green_blur = 5
orange_blur = 27
yellow_blur = 1

# define range of green of retroreflective tape in HSV
lower_green = np.array([40, 75, 75])
upper_green = np.array([96, 255, 255])

lower_yellow = np.array([14, 150, 100])
upper_yellow = np.array([30, 255, 255])

# initialize some variable used later for user input
color_is_yellow = True
lower_color = lower_yellow
upper_color = upper_yellow


switch = 1


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
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

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
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
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
            #print("bounding rec x: " + str(y))
            #print("bounding rec y: " + str(x))
            print("bounding rec height: " + str(h))
            print("bounding rec width: " + str(w))
        
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

                    # Gets the (x, y) and radius of the enclosing circle of contour
                    #(x, y), radius = cv2.minEnclosingCircle(cnt)
                    # Rounds center of enclosing circle
                    #center = (int(x), int(y))
                    # Rounds radius of enclosning circle
                    #radius = int(radius)
                    # Makes bounding rectangle of contour
                    #rx, ry, rw, rh = cv2.boundingRect(cnt)
                    #x, y, w, h = cv2.boundingRect(cnt)

                    # Draws contour of bounding rectangle in red
                    #cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 1)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
                   
                    # Draws circle in cyan
                    #cv2.circle(image, center, radius, (255, 255,0), 1)

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



# Draws Contours and finds center and yaw of vision targets
# centerX is center x coordinate of image
# centerY is center y coordinate of image
def findTape(contours, image, centerX, centerY):
    screenHeight, screenWidth, channels = image.shape
    # Seen vision targets (correct angle, adjacent to each other)
    targets = []

    if len(contours) >= 2:
        # Sort contours by area size (biggest to smallest)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

        cntHeight = 0

        biggestCnts = []
        for cnt in cntsSorted:
            # Get moments of contour; mainly for centroid
            M = cv2.moments(cnt)
            # Get convex hull (bounding polygon on contour)
            hull = cv2.convexHull(cnt)
            # Calculate Contour area
            cntArea = cv2.contourArea(cnt)
            # calculate area of convex hull
            hullArea = cv2.contourArea(hull)

            x, y, w, cntHeight = cv2.boundingRect(cnt)

            pts, dim, a = cv2.minAreaRect(cnt)

            x = pts[0]
            y = pts[1]

            if dim[0] > dim[1]:
                cntHeight = dim[0]
            else:
                cntHeight = dim[1]

            # print("The contour height is, ", cntHeight)
            # Filters contours based off of size
            if (checkContours(cntArea, hullArea)):
                ### MOSTLY DRAWING CODE, BUT CALCULATES IMPORTANT INFO ###
                # Gets the centeroids of contour
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    theCX = cx
                    theCY = cy
                else:
                    cx, cy = 0, 0
                if (len(biggestCnts) < 13):
                    #### CALCULATES ROTATION OF CONTOUR BY FITTING ELLIPSE ##########
                    rotation = getEllipseRotation(image, cnt)

                    # Calculates yaw of contour (horizontal position in degrees)
                    yaw = calculateYaw(cx, centerX, H_FOCAL_LENGTH)
                    # Calculates yaw of contour (horizontal position in degrees)
                    pitch = calculatePitch(cy, centerY, V_FOCAL_LENGTH)
                    # Calculates Distance
                    dist = calculateDistance(1, 2, pitch);

                    ##### DRAWS CONTOUR######
                    # Gets rotated bounding rectangle of contour
                    rect = cv2.minAreaRect(cnt)
                    # Creates box around that rectangle
                    box = cv2.boxPoints(rect)
                    # Not exactly sure
                    box = np.int0(box)
                    # Draws rotated rectangle

                    cv2.drawContours(image, [box], 0, (23, 184, 80), 3)

                    # Calculates yaw of contour (horizontal position in degrees)
                    yaw = calculateYaw(cx, centerX, H_FOCAL_LENGTH)
                    # Calculates yaw of contour (horizontal position in degrees)
                    pitch = calculatePitch(cy, centerY, V_FOCAL_LENGTH)
                    # Calculates Distance
                    dist = calculateDistance(1, 2, pitch);

                    # Draws a vertical white line passing through center of contour
                    cv2.line(image, (cx, screenHeight), (cx, 0), (255, 255, 255))
                    # Draws a white circle at center of contour
                    cv2.circle(image, (cx, cy), 6, (255, 255, 255))

                    # Draws the contours
                    cv2.drawContours(image, [cnt], 0, (23, 184, 80), 1)

                    # Gets the (x, y) and radius of the enclosing circle of contour
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    # Rounds center of enclosing circle
                    center = (int(x), int(y))
                    # Rounds radius of enclosning circle
                    radius = int(radius)
                    # Makes bounding rectangle of contour
                    rx, ry, rw, rh = cv2.boundingRect(cnt)
                    boundingRect = cv2.boundingRect(cnt)
                    # Draws countour of bounding rectangle and enclosing circle in green
                    cv2.rectangle(image, (rx, ry), (rx + rw, ry + rh), (23, 184, 80), 1)

                    cv2.circle(image, center, radius, (23, 184, 80), 1)

                    # Appends important info to array
                    if [cx, cy, rotation, cnt, cntHeight] not in biggestCnts:
                        biggestCnts.append([cx, cy, rotation, cnt, cntHeight])

        # Sorts array based on coordinates (leftmost to rightmost) to make sure contours are adjacent
        biggestCnts = sorted(biggestCnts, key=lambda x: x[0])
        # Target Checking
        for i in range(len(biggestCnts) - 1):
            # Rotation of two adjacent contours
            tilt1 = biggestCnts[i][2]
            tilt2 = biggestCnts[i + 1][2]

            # x coords of contours
            cx1 = biggestCnts[i][0]
            cx2 = biggestCnts[i + 1][0]

            cy1 = biggestCnts[i][1]
            cy2 = biggestCnts[i + 1][1]
            # If contour angles are opposite
            if (np.sign(tilt1) != np.sign(tilt2)):
                centerOfTarget = math.floor((cx1 + cx2) / 2)
                # ellipse negative tilt means rotated to right
                # Note: if using rotated rect (min area rectangle)
                #      negative tilt means rotated to left
                # If left contour rotation is tilted to the left then skip iteration
                if (tilt1 > 0):
                    if (cx1 < cx2):
                        continue
                # If left contour rotation is tilted to the left then skip iteration
                if (tilt2 > 0):
                    if (cx2 < cx1):
                        continue
                # Angle from center of camera to target (what you should pass into gyro)
                yawToTarget = calculateYaw(centerOfTarget, centerX, H_FOCAL_LENGTH)
                pitchToTarget = calculatePitch(theCY, centerY, H_FOCAL_LENGTH)
                # distToTarget = calculateDistance(1, 2, pitchToTarget)
                distToTarget = calculateDistWPILib(biggestCnts[i][4])
                # Make sure no duplicates, then append
                if [centerOfTarget, yawToTarget, distToTarget] not in targets:
                    targets.append([centerOfTarget, yawToTarget, distToTarget])
    # Check if there are targets seen
    if (len(targets) > 0):
        # pushes that it sees vision target to network tables

        # Sorts targets based on x coords to break any angle tie
        targets.sort(key=lambda x: math.fabs(x[0]))
        finalTarget = min(targets, key=lambda x: math.fabs(x[1]))
        # Puts the yaw on screen
        # Draws yaw of target + line where center of target is
        cv2.putText(image, "Yaw: " + str(finalTarget[1]), (40, 40), cv2.FONT_HERSHEY_COMPLEX, .6,
                    (255, 255, 255))
        cv2.putText(image, "Dist: " + str(finalTarget[2]), (40, 90), cv2.FONT_HERSHEY_COMPLEX, .6,
                    (255, 255, 255))
        cv2.line(image, (finalTarget[0], screenHeight), (finalTarget[0], 0), (255, 0, 0), 2)

        currentAngleError = finalTarget[1]
        # pushes vision target angle to network table


    cv2.line(image, (round(centerX), screenHeight), (round(centerX), 0), (255, 255, 255), 2)

    return image


# Finds the balls from the masked image and displays them on original stream + network tables
def findControlPanel(frame, mask):
    # Finds contours
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

# Finds curvature of a set of points
def curvature(pts):
    x = [pts[i][0][0] for i in range(len(pts))]
    y = [pts[i][0][1] for i in range(len(pts))]
    spl = UnivariateSpline(x, y)
    curv = []
    for i in range(len(x)):
        derivs = spl.derivatives(x[i])
        c = abs(derivs[1]*(1 + derivs[0]**2)**-1.5)
        curv.append(c)
    return(curv)


# Finds the balls from the masked image and displays them on original stream + network tables
def findOuterTarget2(frame, mask):

    global real_world_coordinates

    # Gets the shape of video
    screenHeight, screenWidth, _ = frame.shape
    # Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    # Finds contours
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return frame

    # Copies frame and stores it in image
    image = frame.copy()

    cnt = sorted(contours,key=cv2.contourArea, reverse=True)[0]
    print("Number of points in contour: ", len(cnt))
    cv2.drawContours(image, [cnt], -1, purple, 3)

    # Get the left, right, and bottom points
    # extreme points
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])

    # draw extreme points

    # from https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
    cv2.circle(image, leftmost, 3, green, -1)
    cv2.circle(image, rightmost, 3, red, -1)
    #cv2.circle(image, topmost, 3, white, -1)
    cv2.circle(image, bottommost, 3, magenta, -1)
    print('extreme points', leftmost,rightmost,topmost,bottommost)

    # Calculate centroid
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    print('centroid = ',cx,cy)
    cv2.line(image,(cx-10,cy-10),(cx+10,cy+10),red,2)
    cv2.line(image,(cx-10,cy+10),(cx+10,cy-10),red,2)

    # Determine if bottom point is to the left or right of target based on centroid
    bottommost_is_left = False
    if bottommost[0] < cx:
        bottommost_is_left = True
        print("bottommost is on the left")
    else:
        bottommost_is_left = False
        print("bottommost is on the right") 

    # Order of points in contour appears to be top, left, bottom, right

    # Run through all points in the contour, collecting points to build lines whose
    # intersection gives the fourth point.
    topmost_index = leftmost_index = bottommost_index = rightmost_index = -1
    for i in range(len(cnt)):
        point = tuple(cnt[i][0])
        if (point == topmost):
            topmost_index = i
            print("Found topmost:", topmost, " at index ", i)
        if (point == leftmost):
            print("Found leftmost:", leftmost, " at index ", i)
            leftmost_index = i
        if (point == bottommost):
            print("Found bottommost:", bottommost, " at index ", i)
            bottommost_index = i
        if (point == rightmost):
            print("Found rightmost:", rightmost, " at index ", i)
            rightmost_index = i

    if ((topmost_index == -1)   or (leftmost_index == -1) or 
        (rightmost_index == -1) or (bottommost_index == -1)    ):
        print ("Critical point(s) not found in contour")
        return image

    # In some cases, topmost and rightmost pixel will be the same so that index of
    # rightmost pixel in contour will be zero (instead of near the end of the contour)
    # To handle this case correctly and keep the code simple, set index of rightmost
    # pixel to be the final one in the contour. (The corresponding point and the actual
    # rightmost pixel will be very close.) 
    if rightmost_index == 0:
        rightmost_index = len(cnt-1)

    if bottommost_is_left == True:
        # Get set of points after bottommost
        num_points_to_collect = max(int(0.25*(rightmost_index-leftmost_index)), 4)
        print("num_points_to_collect=", num_points_to_collect)
        if num_points_to_collect == 0:
            print ("num_points_to_collect=0, exiting")
            return image
        line1_points = cnt[bottommost_index:bottommost_index+num_points_to_collect+1]
        # Get set of points before rightmost
        num_points_to_collect = max(int(0.25*(bottommost_index-leftmost_index)), 4)
        if num_points_to_collect == 0:
            print ("num_points_to_collect=0, exiting")
            return image
        print("num_points_to_collect=", num_points_to_collect)
        line2_points = cnt[(rightmost_index-num_points_to_collect)%len(cnt):rightmost_index+1]
    else:
        # Get set of points after leftmost
        num_points_to_collect = max(int(0.25*(rightmost_index-bottommost_index)), 4)
        if num_points_to_collect == 0:
            print ("num_points_to_collect=0, exiting")
            return image
        print("num_points_to_collect=", num_points_to_collect)
        line1_points = cnt[leftmost_index:leftmost_index+num_points_to_collect+1]
        # Get set of point before bottommost
        num_points_to_collect = max(int(0.25*(rightmost_index-leftmost_index)), 4)
        if num_points_to_collect == 0:
            print ("num_points_to_collect=0, exiting")
            return image
        print("num_points_to_collect=", num_points_to_collect)
        line2_points = cnt[bottommost_index-num_points_to_collect:bottommost_index+1]

    x1 = [line1_points[i][0][0] for i in range(len(line1_points))]
    y1 = [line1_points[i][0][1] for i in range(len(line1_points))]
    m1, b1, r_value1, p_value1, std_err1 = stats.linregress(x1,y1)
    print("m1=", m1, " b1=", b1)
    [v11,v21,x01,y01] = cv2.fitLine(line1_points, cv2.DIST_L2,0,0.01,0.01)
    m1 = v21/v11
    b1 = y01 - m1*x01
    print("From fitline: m1=", m1, " b1=", b1)

    x2 = [line2_points[i][0][0] for i in range(len(line2_points))]
    y2 = [line2_points[i][0][1] for i in range(len(line2_points))]
    m2, b2, r_value2, p_value2, std_err2 = stats.linregress(x2,y2)
    print("m2=", m2, " b2=", b2)
    [v12,v22,x02,y02] = cv2.fitLine(line2_points, cv2.DIST_L2,0,0.01,0.01)
    m2 = v22/v12
    b2 = y02 - m2*x02
    print("From fitline: m2=", m2, " b2=", b2)

    xint = (b2-b1)/(m1-m2)
    yint = m1*xint+b1
    print("xint=", xint, " yint=", yint)
    int_point = tuple([int(xint), int(yint)])
    print("int_point=", int_point)
    cv2.circle(image, int_point, 3, white, -1)

    # Call solvePnp()
    if bottommost_is_left == True:
        image_points = np.array([
                                 leftmost,
                                 rightmost,
                                 bottommost,
                                 int_point
                                ], dtype="double")
    else:
        image_points = np.array([
                                 leftmost,
                                 rightmost,
                                 int_point,
                                 bottommost
                                ], dtype="double")

    size = [screenHeight, screenWidth]
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]
                             ], dtype = "double")

    camera_matrix = np.array([[676.9, 0, 303.9],
                              [0, 677.96, 226.6],
                              [0, 0, 1]
                             ], dtype = "double")


    print ("Camera Matrix :\n", camera_matrix)

    print("image points:", image_points)
 
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rvec, tvec) = cv2.solvePnP(real_world_coordinates, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_P3P)        

    print ("Rotation Vector:\n", rvec)
    print ("Translation Vector:\n", tvec)

    tilt_deg = 35.0
    tilt_rad = math.radians(tilt_deg)
    x = tvec[0][0]
    z = tvec[1][0]*math.sin(tilt_rad) + tvec[2][0]*math.cos(tilt_rad)
    distance = math.sqrt(z**2 + x**2)
    print("distance [ft]=", distance/12.0)

    # Shows the contours overlayed on the original video
    return image

def get_four_points(cnt):
    # Get the left, right, and bottom points
    # extreme points
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    #print('extreme points', leftmost,rightmost,topmost,bottommost)

    # Calculate centroid
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    #print('centroid = ',cx,cy)
    #cv2.line(image,(cx-10,cy-10),(cx+10,cy+10),red,2)
    #cv2.line(image,(cx-10,cy+10),(cx+10,cy-10),red,2)

    # Determine if bottom point is to the left or right of target based on centroid
    bottommost_is_left = False
    if bottommost[0] < cx:
        bottommost_is_left = True
        #print("bottommost is on the left")
    else:
        bottommost_is_left = False
        #print("bottommost is on the right") 

    # Order of points in contour appears to be top, left, bottom, right

    # Run through all points in the contour, collecting points to build lines whose
    # intersection gives the fourth point.
    topmost_index = leftmost_index = bottommost_index = rightmost_index = -1
    for i in range(len(cnt)):
        point = tuple(cnt[i][0])
        if (point == topmost):
            topmost_index = i
            #print("Found topmost:", topmost, " at index ", i)
        if (point == leftmost):
            #print("Found leftmost:", leftmost, " at index ", i)
            leftmost_index = i
        if (point == bottommost):
            #print("Found bottommost:", bottommost, " at index ", i)
            bottommost_index = i
        if (point == rightmost):
            #print("Found rightmost:", rightmost, " at index ", i)
            rightmost_index = i

    if ((topmost_index == -1)   or (leftmost_index == -1) or 
        (rightmost_index == -1) or (bottommost_index == -1)    ):
        print ("Critical point(s) not found in contour")
        return image

    # In some cases, topmost and rightmost pixel will be the same so that index of
    # rightmost pixel in contour will be zero (instead of near the end of the contour)
    # To handle this case correctly and keep the code simple, set index of rightmost
    # pixel to be the final one in the contour. (The corresponding point and the actual
    # rightmost pixel will be very close.) 
    if rightmost_index == 0:
        rightmost_index = len(cnt-1)

    if bottommost_is_left == True:
        # Get set of points after bottommost
        num_points_to_collect = max(int(0.25*(rightmost_index-leftmost_index)), 4)
        #print("num_points_to_collect=", num_points_to_collect)
        if num_points_to_collect == 0:
            print ("num_points_to_collect=0, exiting")
            return image
        line1_points = cnt[bottommost_index:bottommost_index+num_points_to_collect+1]
        # Get set of points before rightmost
        num_points_to_collect = max(int(0.25*(bottommost_index-leftmost_index)), 4)
        if num_points_to_collect == 0:
            print ("num_points_to_collect=0, exiting")
            return image
        #print("num_points_to_collect=", num_points_to_collect)
        line2_points = cnt[(rightmost_index-num_points_to_collect)%len(cnt):rightmost_index+1]
    else:
        # Get set of points after leftmost
        num_points_to_collect = max(int(0.25*(rightmost_index-bottommost_index)), 4)
        if num_points_to_collect == 0:
            print ("num_points_to_collect=0, exiting")
            return image
        #print("num_points_to_collect=", num_points_to_collect)
        line1_points = cnt[leftmost_index:leftmost_index+num_points_to_collect+1]
        # Get set of point before bottommost
        num_points_to_collect = max(int(0.25*(rightmost_index-leftmost_index)), 4)
        if num_points_to_collect == 0:
            print ("num_points_to_collect=0, exiting")
            return image
        #print("num_points_to_collect=", num_points_to_collect)
        line2_points = cnt[bottommost_index-num_points_to_collect:bottommost_index+1]


    min_points_for_line_fit = 5

    #x1 = [line1_points[i][0][0] for i in range(len(line1_points))]
    #y1 = [line1_points[i][0][1] for i in range(len(line1_points))]
    #m1, b1, r_value1, p_value1, std_err1 = stats.linregress(x1,y1)
    #print("m1=", m1, " b1=", b1)

    if len(line1_points) < min_points_for_line_fit:
        return False, np.zeros(4,1) 

    [v11,v21,x01,y01] = cv2.fitLine(line1_points, cv2.DIST_L2,0,0.01,0.01)
    if (v11==0):
        print("Warning v11=0")
        v11 = 0.1
    m1 = v21/v11
    b1 = y01 - m1*x01
    #print("From fitline: m1=", m1, " b1=", b1)

    #x2 = [line2_points[i][0][0] for i in range(len(line2_points))]
    #y2 = [line2_points[i][0][1] for i in range(len(line2_points))]
    #m2, b2, r_value2, p_value2, std_err2 = stats.linregress(x2,y2)
    #print("m2=", m2, " b2=", b2)

    if len(line2_points) < min_points_for_line_fit:
        return False, np.zeros(4,1) 

    [v12,v22,x02,y02] = cv2.fitLine(line2_points, cv2.DIST_L2,0,0.01,0.01)
    m2 = v22/v12
    if (v12==0):
        print("Warning v11=0")
        v12 = 0.1
    b2 = y02 - m2*x02
    #print("From fitline: m2=", m2, " b2=", b2)

    if (m1 == m2):
        return False, np.zeros(4,1) 
    xint = (b2-b1)/(m1-m2)
    yint = m1*xint+b1
    #print("xint=", xint, " yint=", yint)
    int_point = tuple([int(xint), int(yint)])

    if bottommost_is_left == True:
        four_points = np.array([
                                 leftmost,
                                 rightmost,
                                 bottommost,
                                 int_point
                                ], dtype="double")
    else:
        four_points = np.array([
                                 leftmost,
                                 rightmost,
                                 int_point,
                                 bottommost
                                ], dtype="double")

    return four_points

# Finds the balls from the masked image and displays them on original stream + network tables
def findOuterTarget(frame, mask):

    global real_world_coordinates

    # Gets the shape of video
    screenHeight, screenWidth, _ = frame.shape
    # Gets center of height and width
    centerX = (screenWidth / 2) - .5
    centerY = (screenHeight / 2) - .5
    # Finds contours
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return frame

    # Copies frame and stores it in image
    image = frame.copy()

    cnt = sorted(contours,key=cv2.contourArea, reverse=True)[0]
    print("Number of points in contour: ", len(cnt))
    cv2.drawContours(image, [cnt], -1, purple, 3)

    image_points = get_four_points(cnt)

    size = [screenHeight, screenWidth]
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]
                             ], dtype = "double")

    camera_matrix = np.array([[676.9, 0, 303.9],
                              [0, 677.96, 226.6],
                              [0, 0, 1]
                             ], dtype = "double")


    print ("Camera Matrix :\n", camera_matrix)

    print("image points:", image_points)
 
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rvec, tvec) = cv2.solvePnP(real_world_coordinates, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_P3P)
 
    if ((success == False) or (len(tvec) < 3)):
        print ("solvePnp() failed")
        return image;

    print ("Rotation Vector:\n", rvec)
    print ("Translation Vector:\n", tvec)

    tilt_deg = 35.0
    tilt_rad = math.radians(tilt_deg)
    x = tvec[0][0]
    z = tvec[1][0]*math.sin(tilt_rad) + tvec[2][0]*math.cos(tilt_rad)
    distance = math.sqrt(z**2 + x**2)
    print("distance [ft]=", distance/12.0)

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
        print(x, y, img[y, x, 2], img[y, x, 1], img[y, x, 0], cv2.cvtColor(green,cv2.COLOR_BGR2HSV))


Driver = False
Tape = False
PowerCell = False
ControlPanel = False
OuterTarget = True


img = images[0]

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
            elif OuterTarget:
                color_is_yellow = False
                lower_color = lower_green
                upper_color = upper_green
                boxBlur = blurImg(frame, green_blur)
                threshold = threshold_video(lower_green, upper_green, boxBlur)
                processed = findOuterTarget(frame, threshold)

    cv2.imshow(filenames[currentImg] + " raw", img)
    cv2.imshow(filenames[currentImg] + " threshold", threshold)
    cv2.imshow(filenames[currentImg] + " processed", processed)
    cv2.setMouseCallback('raw', draw_circle)

    key = cv2.waitKey(0)
    print(key) 

    if key == 27: # <ESC> user wants to exit
        break
    elif key == 112: # 'p' previous image
        if currentImg - 1 < 0:
            currentImg = imgLength - 1
        else:
            currentImg = currentImg - 1
    elif key == 110: # 'n' next image
        if currentImg + 1 > imgLength - 1:
            currentImg = 0
        else:
            currentImg = currentImg + 1

    elif key == 104: # 'h' decrease lower hue
        if (lower_color[0] > 0):
            lower_color[0] = lower_color[0] - 1;
        print()
        print('Lower hue: ', lower_color[0])
    elif key == 72: # 'H' increase lower hue
        if (lower_color[0] < upper_color[0]):
            lower_color[0] = lower_color[0] + 1;
        print()
        print('Lower hue: ', lower_color[0])
    elif key == 106: # 'j' decrease upper hue
        if (upper_color[0] > lower_color[0]):
            upper_color[0] = upper_color[0] - 1;
        print()
        print('Upper hue: ', upper_color[0])
    elif key == 74: # 'J' increase upper hue
        if (upper_color[0] < 255):
            upper_color[0] = upper_color[0] + 1;
        print()
        print('Upper hue: ', upper_color[0])


    elif key == 115: # 's' decrease lower saturation
        if (lower_color[1] > 0):
            lower_color[1] = lower_color[1] - 1;
        print()
        print('Lower saturation: ', lower_color[1])
    elif key == 83: # 'S' increase lower saturation
        if (lower_color[1] < upper_color[1]):
            lower_color[1] = lower_color[1] + 1;
        print()
        print('Lower saturation: ', lower_color[1])
    elif key == 100: # 'd' decrease upper saturation
        if (upper_color[1] > lower_color[1]):
            upper_color[1] = upper_color[1] - 1;
        print()
        print('Upper saturation: ', upper_color[1])
    elif key == 68: # 'D' increase upper saturation
        if (upper_color[1] < 255):
            upper_color[1] = upper_color[1] + 1;
        print()
        print('Upper saturation: ', upper_color[1])

    elif key == 118: # 'v' decrease lower hue value
        if (lower_color[2] > 0):
            lower_color[2] = lower_color[2] - 1;
        print()
        print('Lower value: ', lower_color[2])
    elif key == 86: # 'V' increase lower hue value
        if (lower_color[2] < upper_color[2]):
            lower_color[2] = lower_color[2] + 1;
        print()
        print('Lower value: ', lower_color[2])
    elif key == 98: # 'b' decrease upper hue value
        if (upper_color[2] > lower_color[2]):
            upper_color[2] = upper_color[2] - 1;
        print()
        print('Upper value: ', upper_color[2])
    elif key == 66: # 'B' increase upper hue value
        if (upper_color[2] < 255):
            upper_color[2] = upper_color[2] + 1
        print()
        print('Upper value: ', upper_color[2])

    elif key == 109: # 'm' print hue, saturation, value bounds for mask
        print()
        print('Color bounds for ')
        if color_is_yellow == True:
            print('YELLOW:')
        else:
            print('GREEN:')
        print('Hue:        [', lower_color[0], ',', upper_color[1], ']')
        print('Saturation: [', lower_color[1], ',', upper_color[1], ']')
        print('Value:      [', lower_color[2], ',', upper_color[2], ']')

    elif key == 99: # 'c' toogle between yellow and green
        color_is_yellow = not(color_is_yellow)
        if color_is_yellow == True: # yellow
            lower_color = lower_yellow
            upper_color = upper_yellow
        else: # green
            lower_color = lower_green
            upper_color = upper_green
        print()
        print('color_is_yellow: ', color_is_yellow)


    elif key == 107: # 'k'
        #intMaskMethod = 1
        print()
        print('To be implemented')
    elif key == 109: # 'm'
        #intMaskMethod = 2
        print()
        print('To be implemented')
    elif key == 32: # space
        print()
        print('...repeat...')
    else:
        print ("Unrecognized key: ", key)



    if (currentImg == imgLength):
         currentImg = 0

    img = images[currentImg]

    cv2.destroyAllWindows()





