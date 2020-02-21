# This is a pseudo code file for Merge Robotics, 2020, Infinite Recharge
# This is task I - > OpenCV "Contours Continued."  Not sure if it is clear by now, 
# but OpenCV can do a lot of things, we need to understand what it offers to complete 
# our vision code.  For a given single contour, (meaning it was imaged and masked and 
# converted to a coordinate array), you need to be able to use a number of OpenCV functions.
# Please experiment with the following, easiest is to simply draw them back to a blank image
# or on top of original.

# - contour perimeter, contour approximation, bounding rectangles, 
# minimum enclosing circle, fitting elipse, fitting line, aspect ratio
# extent, solidity, equivalent diameter, orientation, points, min/max
# mean color, extreme points

# useful links
# https://docs.opencv.org/3.4.7/dd/d49/tutorial_py_contour_features.html
# https://docs.opencv.org/3.4.7/d1/d32/tutorial_py_contour_properties.html

import numpy as np
import cv2
import sys
import os
import math
import datetime
import time

import json
import random
import sys

from pathlib import Path
from visual4_old import get_four

print("Using python version {0}".format(sys.version))
print('OpenCV Version = ', cv2.__version__)
print()

# define colors
purple = (165, 0, 120)
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
cyan = (252, 252, 3)
magenta = (252, 3, 252)
yellow = (3, 252, 252)
black = (0, 0, 0)
white = (252, 252, 252)
orange = (3, 64, 252) 

minAreaContour = 200

# from https://stackoverflow.com/questions/41462419/python-slope-given-two-points-find-the-slope-answer-works-doesnt-work/41462583
def get_slope(x1, y1, x2, y2):
    return (y2-y1)/(x2-x1) 

# homemade! http://home.windstream.net/okrebs/Ch9-1.gif
def get_opposit(hyp, theta):
    return hyp*math.sin(math.radians(theta))

# homemade! http://home.windstream.net/okrebs/Ch9-1.gif
def get_adjacent(hyp, theta):
    return abs(hyp*math.cos(math.radians(theta)))

def milliSince1970():
    return int((datetime.datetime.utcnow() - datetime.datetime(1970,1,1)).total_seconds()*1000)               

#Simple function that displays 4 corners on an image
#A np.array() is expected as the input argument
def displaycorners(image, outer_corners):
    # draw extreme points
    # from https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
    if len(outer_corners) == 4: #this is methods 1 to 4 
        cv2.circle(image, (int(outer_corners[0,0]),int(outer_corners[0,1])), 6, green, -1)
        cv2.circle(image, (int(outer_corners[1,0]),int(outer_corners[1,1])), 6, red, -1)
        cv2.circle(image, (int(outer_corners[2,0]),int(outer_corners[2,1])), 6, white,-1)
        cv2.circle(image, (int(outer_corners[3,0]),int(outer_corners[3,1])), 6, blue, -1)
        #print('extreme points', leftmost,rightmost,topmost,bottommost)
    else: # this assumes len is 5 and method 5
        cv2.circle(image, (int(outer_corners[0,0]),int(outer_corners[0,1])), 6, green, -1)
        cv2.circle(image, (int(outer_corners[1,0]),int(outer_corners[1,1])), 6, blue, -1)
        cv2.circle(image, (int(outer_corners[2,0]),int(outer_corners[2,1])), 6, purple, -1)
        cv2.circle(image, (int(outer_corners[3,0]),int(outer_corners[3,1])), 6, white,-1)
        cv2.circle(image, (int(outer_corners[4,0]),int(outer_corners[4,1])), 6, red, -1)


# select folder of interest
posCodePath = Path(__file__).absolute()
strVisionRoot = posCodePath.parent.parent
#strImageFolder = str(strVisionRoot / 'OuterTargetFullDistance')
strImageFolder = str(strVisionRoot / 'OuterTargetFullScale')
#strImageFolder = str(strVisionRoot / 'OuterTargetSketchup')
#strImageFolder = str(strVisionRoot / 'OuterTargetHalfScale')
#strImageFolder = str(strVisionRoot / 'OuterTargetImages')
#strImageFolder = str(strVisionRoot / 'OuterTargetLiger')
#strImageFolder = str(strVisionRoot / 'OuterTargetRingTest')
#strImageFolder = str(strVisionRoot / 'OuterTargetProblems')

print (strImageFolder)
booBlankUpper = False

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
    ##print (i, ' ', strImageInput)
    print ()
    print (photos[i])

    ## read file
    imgImageInput = cv2.imread(strImageInput)
    intImageHeight, intImageWidth = imgImageInput.shape[:2]

    if booBlankUpper:
        ## blank upper portion from Task K
        cv2.rectangle(imgImageInput, (0,0), (intImageWidth, int(intImageHeight/2-10)), black, -1)

    #cv2.imshow('imgImageInput', imgImageInput)
    #cv2.moveWindow('imgImageInput',300,350)

    # start
    start = milliSince1970()

    # Convert BGR to HSV
    hsvImageInput = cv2.cvtColor(imgImageInput, cv2.COLOR_BGR2HSV)

    # define range of green color in HSV
    lower_color = np.array([55,55,55])
    upper_color = np.array([100,255,255])

    # Threshold the HSV image to get only green colors
    binary_mask = cv2.inRange(hsvImageInput, lower_color, upper_color)

    # mask the image to only show green or green images
    # Bitwise-AND mask and original image
    green_mask = cv2.bitwise_and(imgImageInput, imgImageInput, mask=binary_mask)
    #cv2.imwrite('green_mask.jpg',green_mask)

    # display the masked images to screen
    #cv2.imshow('hsvImageInput', hsvImageInput)
    #cv2.moveWindow('hsvImageInput',10,50)

    #cv2.imshow('binary_mask',binary_mask)
    cv2.imshow('green_masked',green_mask)
    #cv2.moveWindow('green_masked',400,550)
    cv2.moveWindow('green_masked',20,50) 
      
    # generate the contours and display
    imgFindContourReturn, contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imgContours = green_mask.copy()

    # sort contours by area descending
    initialSortedContours = sorted(contours, key = cv2.contourArea, reverse = True)[:15]

    cv2.drawContours(imgContours, initialSortedContours, -1, yellow, 1)
    print('Found ', len(contours), 'contours in image')
    #print (contours)

    initialFilteredContours = []

    if initialSortedContours:

        for (j, indiv) in enumerate(initialSortedContours):

            print('indiv', j)

            # Area
            area = cv2.contourArea(indiv)
            print('area = ', area)
            if area < minAreaContour: continue

            # rotated rectangle
            rect = cv2.minAreaRect(indiv)
            print('rotated rectangle = ',rect)
            (xr,yr),(wr,hr),ar = rect
            #### more research https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned
            if hr > wr:
                ar = ar + 90
                wr, hr = [hr, wr]
            else:
                ar = ar + 180
            if ar == 180:
                ar = 0

            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(imgContours,[box],0,blue,2)
            minAAspect = float(wr)/hr
            minAextent = float(area)/(wr*hr) 
            print('minimum area rectangle aspect = ', minAAspect)
            print('minimum area rectangle extent = ', minAextent)

            if (minAextent < 0.16 or minAextent > 0.26): continue
            if (minAAspect < 2.0 or minAAspect > 3.0): continue

            # Hull
            hull = cv2.convexHull(indiv)
            #cv2.drawContours(imgContours, [hull], -1, orange, 5)
            #cv2.imshow('hull over green mask', imgContours)
            hull_area = cv2.contourArea(hull)
            print('area of convex hull',hull_area)
            solidity = float(area)/hull_area
            print('solidity from convex hull', float(area)/hull_area)

            if (solidity < 0.22 or solidity > 0.30): continue

            initialFilteredContours.append(indiv)

    if initialFilteredContours:

        cnt = initialFilteredContours[0]
        print('original contour length = ', len(cnt))
        cv2.drawContours(imgContours, [cnt], -1, purple, 3)

        # Area
        area = cv2.contourArea(cnt)
        print('area = ', area)
        
        # Perimeter
        perimeter = cv2.arcLength(cnt,True)
        print('perimeter = ', perimeter)

        # Hull
        hull = cv2.convexHull(cnt)
        #print('hull', hull)
        print('hull contour length = ', len(hull))
        hull_mask = np.zeros([intImageHeight,intImageWidth,1],dtype=np.uint8)
        cv2.drawContours(hull_mask, [hull], -1, 255, cv2.FILLED)
        #cv2.imshow('hull over green mask', imgContours)
        hull_area = cv2.contourArea(hull)
        print('area of convex hull',hull_area)
        print('solidity from convex hull', float(area)/hull_area)

        # Check Convexity
        print('convexity is', cv2.isContourConvex(cnt))

        # straight bounding rectangle
        xb,yb,wb,hb = cv2.boundingRect(cnt)
        print('straight bounding rectangle = ', (xb,yb) ,wb,hb)
        bounding_rect = (xb,yb,wb,hb)
        #cv2.rectangle(imgContours,(xb,yb),(xb+wb,yb+hb),green,2)
        print('bounding rectangle aspect = ', float(wb)/float(hb))
        print('bounding rectangle extent = ', float(area)/(float(wb)*float(hb)))

        # rotated rectangle
        rect = cv2.minAreaRect(cnt)
        print('rotated rectangle = ',rect)
        (xr,yr),(wr,hr),ar = rect
        #### more research https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned
        if hr > wr:
            ar = ar + 90
            wr, hr = [hr, wr]
        else:
            ar = ar + 180
        if ar == 180:
            ar = 0

        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(imgContours,[box],0,blue,2)
        print('minimum area rectangle aspect = ', float(wr)/hr)
        print('minimum area rectangle extent = ', float(area)/(wr*hr))

        # Moment and Centroid
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        print('centroid = ',cx,cy)
        ct = int(hr/12)
        cv2.line(imgContours,(cx-ct,cy-ct),(cx+ct,cy+ct),red,2)
        cv2.line(imgContours,(cx-ct,cy+ct),(cx+ct,cy-ct),red,2)

        # minimum enclosing circle
        (xc,yc),radius = cv2.minEnclosingCircle(cnt)
        print('minimum enclosing circle = ', (xc,yc),radius)
        center = (int(xc),int(yc))
        radius = int(radius)
        #cv2.circle(imgContours,center,radius,green,2)
        #equi_diameter = np.sqrt(4*area/np.pi)
        #cv2.circle(imgContours, (cx,cy), int(equi_diameter/2), purple, 3)

        if len(cnt) > 5:
            # fitting an elipse
            ellipse = cv2.fitEllipse(cnt)
            #print(ellipse)
            # search ellipse to find it return a rotated rectangle in which the ellipse fits
            (xe,ye),(majAxis,minAxis),ae = ellipse
            print('ellipse center, maj axis, min axis, rotation = ', (xe,ye) ,(majAxis, minAxis), ae)
            # search major and minor axis from ellipse
            # https://namkeenman.wordpress.com/2015/12/21/opencv-determine-orientation-of-ellipserotatedrect-in-fitellipse/
            #cv2.ellipse(imgContours,ellipse,orange,2)
            print('ellipse aspect = ', float(majAxis)/minAxis)

        # fitting a line
        rows,cols = binary_mask.shape[:2]
        #[vx,vy,xl,yl] = cv.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01) #errors in VS Code, search online and found fix
        [vx,vy,xl,yl] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
        lefty = int((-xl*vy/vx) + yl)
        righty = int(((cols-xl)*vy/vx)+yl)
        #cv2.line(imgContours,(cols-1,righty),(0,lefty),green,2)
        # http://ottonello.gitlab.io/selfdriving/nanodegree/python/line%20detection/2016/12/18/extrapolating_lines.html
        slope = vy / vx
        intercept = yl - (slope * xl)
        print('fitLine y = ', slope, '* x + ', intercept)

        # aspect ratio
        # added retroactively to bounding, min area and elipse

        # extent calculation
        # added retroactively to bounding and min area

        # solidity
        # added retroactively to the hull

        # equivalent diameter
        # added retroactively to the enclosing circle

        # orientation
        # tweaked ellipse above to reflect details in link

        # mask and pixel points
        # skipping this one...

        # Maximum Value, Minimum Value and their locations of a binary mask not contour!
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(binary_mask)
        print('min_val = ', min_val)
        print('max_val = ', max_val)
        print('min_loc = ', min_loc)
        print('max_loc = ', max_loc)

        # Mean Color or Mean Intensity 
        mean_val1 = cv2.mean(imgImageInput)
        print('mean value from input image = ', mean_val1)
        mean_val2 = cv2.mean(hsvImageInput, mask = binary_mask)
        print('mean value from HSV and mask = ', mean_val2)
        # look at the result of mean_val2 on colorizer.org
        mean_val3 = cv2.mean(green_mask)
        print('mean value from colored mask = ', mean_val3)

        # extreme points
        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        #topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        #bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
        # draw extreme points
        # from https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
        cv2.circle(imgContours, leftmost, 12, green, -1)
        cv2.circle(imgContours, rightmost, 12, red, -1)
        #cv2.circle(imgContours, topmost, 12, white, -1)
        #cv2.circle(imgContours, bottommost, 12, blue, -1)
        #print('extreme points = left',leftmost,'right',rightmost,'top',topmost,'bottom',bottommost)
        print('extreme points = left',leftmost,'right',rightmost)

        # Start of visual4
        # prepare chosen contour for 4 point finder as ROI
        #ROI_mask = binary_mask[yb:yb+hb, xb:xb+wb]
        #intROMHeight, intROMWidth = ROI_mask.shape[:2]
        #imgFindContourReturn, ROIcontours, hierarchy = cv2.findContours(ROI_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #ROISortedContours = sorted(ROIcontours, key = cv2.contourArea, reverse = True)[:1]

        # send chosen contour to 4 point finder, get back found points or None
        #try_get_four = get_four(bounding_rect, intROMWidth, intROMHeight, ROISortedContours[0])
        #if try_get_four is None:
        #    pass
        #else:
        #    only_four = ((try_get_four[0]),(try_get_four[1]),(try_get_four[3]),(try_get_four[4]))
        #    outer_corners = np.array(only_four)
        #    displaycorners(imgContours, outer_corners)
        # End of visual4

        # Start of Method10
        # Do left sloped line first
        left_ys = int(round(leftmost[1]+hr/10,0))
        left_yf = int(round(leftmost[1]+hr,0))
        left_xs = xb+int(round(wr/20,0))
        left_xf = xb+int(round(wr/7,0))
        left_box = [(left_xs, left_ys), (left_xf, left_yf)]
        ROI_mask_left = hull_mask[left_ys:left_yf, left_xs:left_xf]
        cv2.imshow('ROI_mask_left', ROI_mask_left)
        cv2.moveWindow('ROI_mask_left',50,50)
        cv2.rectangle(imgContours, (left_xs, left_ys), (left_xf, left_yf), yellow, 1)
        imgFindContourReturn, leftContour, hierarchy = cv2.findContours(ROI_mask_left, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sortedContours = sorted(leftContour, key = cv2.contourArea, reverse = True)[:1]
        # extract "bottommost-left and bottommost-right" or minX,mayY and maxX,maxY
        # from two points define left line

        # Do right sloped line second
        right_ys = int(round(rightmost[1]+hr/10,0))
        right_yf = int(round(rightmost[1]+hr,0))
        right_xs = xb+wb-int(round(wr/7,0))
        right_xf = xb+wb-int(round(wr/20,0))
        right_box = [(right_xs, right_ys), (right_xf, right_yf)]
        ROI_mask_right = hull_mask[right_ys:right_yf, right_xs:right_xf]
        cv2.imshow('ROI_mask_right', ROI_mask_right)
        cv2.moveWindow('ROI_mask_right',100,50)
        cv2.rectangle(imgContours, (right_xs, right_ys), (right_xf, right_yf), yellow, 1)
        imgFindContourReturn, rightContour, hierarchy = cv2.findContours(ROI_mask_left, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sortedContours = sorted(rightContour, key = cv2.contourArea, reverse = True)[:1]
        # extract "bottommost-left and bottommost-right" or minX,mayY and maxX,maxY
        # from two points define right line

        # Do bottom line third
        center_ys = int(round(yr,0))
        center_yf = int(round(yr+hr,0))
        center_xs = int(round(xr-wr/7,0))
        center_xf = int(round(xr+wr/7,0))
        center_box = [(center_xs, center_ys), (center_xf, center_yf)]
        ROI_mask_center = hull_mask[center_ys:center_yf, center_xs:center_xf]
        cv2.imshow('ROI_mask_center', ROI_mask_center)
        cv2.moveWindow('ROI_mask_center',150,50)
        cv2.rectangle(imgContours, (center_xs, center_ys), (center_xf, center_yf), yellow, 1)
        imgFindContourReturn, centerContour, hierarchy = cv2.findContours(ROI_mask_left, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sortedContours = sorted(centerContour, key = cv2.contourArea, reverse = True)[:1]
        # extract "bottommost-left and bottommost-right" or minX,mayY and maxX,maxY
        # from two points define center line

        # from three lines define two points
        # continue with other method

    stop = milliSince1970()
    # because we are timing in this file, have to add the fps to image processed 
    cv2.putText(imgContours, "elapsed time: " + str(int(stop-start)) + " ms", (40, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6 ,white)
    cv2.putText(imgContours, 'FPS: {:.1f}'.format(1000/(stop-start)), (40, 80), cv2.FONT_HERSHEY_COMPLEX, 0.6 ,white)

    # Display the contours and maths generated
    cv2.imshow('contours and math over green mask', imgContours)
    cv2.moveWindow('contours and math over green mask',650,50)

    ## loop for user input to close - loop indent 2
    booReqToExit = False # true when user wants to exit

    while (True):

    ## wait for user to press key
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
        elif k == 115:
            intMaskMethod = 0
            print()
            print('Mask Method s = Simple In-Range')
            break
        elif k == 107:
            intMaskMethod = 1
            print()
            print('Mask Method k = Knoxville Method')
            break
        elif k == 109:
            intMaskMethod = 2
            print()
            print('Mask Method m = Merge Mystery Method')
            break
        elif k == 32:
            print()
            print('...repeat...')
            break
        else:
            #print (k)
            pass
        ### end of loop indent 2

    ## test for exit main loop request from user
    if booReqToExit:
        break

    ## not exiting, close windows before loading next
    cv2.destroyAllWindows()

# end of main loop indent 1

# cleanup and exit
cv2.destroyAllWindows()