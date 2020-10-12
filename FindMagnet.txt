import numpy as np
import cv2
import sys
from pathlib import Path

def findMagnet(imgImageInput):

    # define colors for code readablility
    purple = (165, 0, 120)
    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)

    # Convert BGR to HSV
    hsvImageInput = cv2.cvtColor(imgImageInput, cv2.COLOR_BGR2HSV)

    # define range of yellow color in HSV
    #28 44 48 is actual square
    #lower_yellow = np.array([26,30,30]) 
    #upper_yellow = np.array([32,255,255])
    #green is 178 100 99 - halved = 89 50 49
    lower_green = np.array([80,220,220]) 
    upper_green = np.array([100,255,255])

    # Threshold the HSV image to get only yellow colors
    binary_mask = cv2.inRange(hsvImageInput, lower_green, upper_green)

    # mask the image to only show yellow or green images
    # Bitwise-AND mask and original image
    yellow_mask = cv2.bitwise_and(imgImageInput, imgImageInput, mask=binary_mask)

    # display the masked images to screen
    #cv2.imshow('hsvImageInput', hsvImageInput)
    cv2.imshow('binary_mask',binary_mask)
    #cv2.imshow('yellow_masked',yellow_mask)
    #cv2.imshow('og image',imgImageInput)

    # generate the contours and display
    imgFindContourReturn, contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imgContours = yellow_mask.copy()
    cv2.drawContours(imgContours, contours, -1, purple, 10)
    print('Found ', len(contours), 'contours in image')

    if len(contours) == 0:
        print('no contours')
        return False, -1, -1
    
    # Moment and Centroid
    cnt = contours[0]
    #print(cnt)
    #print('original',len(cnt),cnt)
    print('original contour length = ', len(cnt))
    M = cv2.moments(cnt)
    #print( M )
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        print('centroid = ',cx,cy)
        cv2.line(imgContours,(cx-10,cy-10),(cx+10,cy+10),red,2)
        cv2.line(imgContours,(cx-10,cy+10),(cx+10,cy-10),red,2)

    cv2.drawContours(imgContours, cnt, -1, purple, 10)
    #cv2.imshow('contours', imgContours)

    # Area
    area = cv2.contourArea(cnt)
    print('area = ', area)

    # Perimeter
    perimeter = cv2.arcLength(cnt,True)
    print('perimeter = ', perimeter)

    # Contour Approximation
    epsilon = 0.005*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    #print('approx', approx)
    #cv2.drawContours(imgContours, approx, -1, red, 10)
    print('approx contour length = ', len(approx))
    #cv2.imshow('approx over yellow mask', imgContours)

    # Hull
    hull = cv2.convexHull(cnt)
    #print('hull', hull)
    print('hull contour length = ', len(hull))
    cv2.drawContours(imgContours, hull, -1, red, 10)
    #cv2.imshow('hull over yellow mask', imgContours)
    hull_area = cv2.contourArea(hull)
    if hull_area != 0:
        print('solidity from convex hull', float(area)/hull_area)

    # Check Convexity
    print('convexity is', cv2.isContourConvex(cnt))

    # straight bounding rectangle
    x,y,w,h = cv2.boundingRect(cnt)
    print('straight bounding rectangle = ', (x,y) ,w,h)
    cv2.rectangle(imgContours,(x,y),(x+w,y+h),green,2)
    print('bounding rectangle aspect = ', float(w)/h)
    print('bounding rectangle extend = ', float(area)/(w*h))

    # rotated rectangle
    rect = cv2.minAreaRect(cnt)
    print('rotated rectangle = ',rect)
    (x,y),(width,height),angleofrotation = rect
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(imgContours,[box],0,blue,2)
    if height != 0:
        print('minimum area rectangle aspect = ', float(width)/height)
        print('minimum area rectangle extent = ', float(area)/(width*height))



    
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
    mean_val3 = cv2.mean(yellow_mask)
    print('mean value from colored mask = ', mean_val3)
    
    # extreme points
    leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
    print('the leftmost point is: ', leftmost)

    rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
    topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
    # draw extreme points
    # from https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
    cv2.circle(imgContours, leftmost, 12, (0, 0, 255), -1)
    cv2.circle(imgContours, rightmost, 12, (0, 255, 0), -1)
    cv2.circle(imgContours, topmost, 12, (255, 0, 0), -1)
    cv2.circle(imgContours, bottommost, 12, (255, 255, 0), -1)
    print('extreme points', leftmost,rightmost,topmost,bottommost)

    # Display the contours and maths generated
    #cv2.imshow('contours and math over yellow mask', imgContours)

    # wait for user input to close
    #k = cv2.waitKey(0)

    # cleanup and exit
    #cv2.destroyAllWindows()

    return True, leftmost, rightmost # to make it return the leftmost coordinate 