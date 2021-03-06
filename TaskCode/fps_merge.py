#
# https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
# import the necessary packages

import argparse
import cv2
import importlib
import numpy as np

from pathlib import Path
from threading import Thread

from visual4_old import get_four
from adrian_pyimage_old import FPS, AdrianVideoStream
#from cscore import CameraServer, VideoSource
from piVideoBits import WebcamVideoStream, startCamera, VideoShow

booWebCam = False
booThreaded = False
booCSCore = False

num_frames = 1000
display = 1

# from MergeFRCPipeline
global currentCam, webcam, cameras, streams, cameraServer, cap, image_width, image_height, prevCam

# setup various variables used
currentCam = 0
cameraConfigs = []
cameras = []
streams = []
image_width = 480
image_height = 640

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

# select folder of interest
posCodePath = Path(__file__).absolute()
strVisionRoot = posCodePath.parent.parent

#strImageFolder = str(strVisionRoot / 'OuterTargetHalfDistance')
#strImageFolder = str(strVisionRoot / 'OuterTargetSketchup')
#strImageFolder = str(strVisionRoot / 'OuterTargetHalfScale')
#strImageFolder = str(strVisionRoot / 'OuterTargetImages')
#strImageFolder = str(strVisionRoot / 'OuterTargetLiger')
#strImageFolder = str(strVisionRoot / 'OuterTargetFullDistance')
strImageFolder = str(strVisionRoot / 'OuterTargetFullScale')

strImageInput = strImageFolder + '/' + 'outer+120f+295d.jpg'

if booWebCam:
    if booThreaded and not booCSCore:
        # created a *threaded* video stream, allow the camera sensor to warmup,
        # and start the FPS counter
        print('[INFO] sampling', num_frames,'THREADED frames from webcam...')
        vs = AdrianVideoStream(src=0).start()   

    elif booThreaded and booCSCore:
        # this is copied from MergeFRCPipeline
        # start cameras
        for cameraConfig in cameraConfigs:
            cs, cameraCapture = startCamera(cameraConfig)
            streams.append(cs)
            cameras.append(cameraCapture)

        # Get the first camera
        webcam = cameras[currentCam]
        cameraServer = streams[currentCam]

        # Start thread reading camera
        cap = WebcamVideoStream(webcam, cameraServer, image_width, image_height).start()
        # cap = cap.findTape
        # (optional) Setup a CvSource. This will send images back to the Dashboard

    else:
        # grab a pointer to the video stream and initialize the FPS counter
        print('[INFO] sampling', num_frames,'frames from file somewhere...')
        stream = cv2.VideoCapture(0)
else:
    print('[INFO] sampling', num_frames,'THREADED frames from file...')

# Allocating new images is very expensive, always try to preallocate
imgImageInput = np.zeros(shape=(image_height, image_width, 3), dtype=np.uint8)

if booThreaded and booCSCore:
    # Start thread outputing stream
    streamViewer = VideoShow(image_width, image_height, cameraServer, frame=imgImageInput).start()

fps = FPS().start()

# loop over some frames
while fps._numFrames < num_frames:

    #
    if booWebCam:
        if booThreaded and not booCSCore:
            # grab the frame from the threaded video stream
            imgImageInput = vs.read()

        elif booThreaded and booCSCore:
            pass
        else:
            # grab the frame from the stream
            (grabbed, imgImageInput) = stream.read()
    else:
        imgImageInput = cv2.imread(strImageInput)
        
    # process start
    cv2.imshow('input stream', imgImageInput)
    cv2.moveWindow('input stream',10,50)

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
    #cv2.moveWindow('green_masked',400,550)

    #cv2.imshow('green_masked',green_mask)
    #cv2.moveWindow('green_masked',20,50) 
      
    # generate the contours and display
    imgFindContourReturn, contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    imgContours = green_mask.copy()

    cv2.drawContours(imgContours, contours, -1, yellow, 1)
    #print('Found ', len(contours), 'contours in image')
    #print (contours)

    # sort contours by area descending
    initialSortedContours = sorted(contours, key = cv2.contourArea, reverse = True)[:3]

    if initialSortedContours:

        indiv = initialSortedContours[0]
        #print('original contour length = ', len(cnt))
        cv2.drawContours(imgContours, [indiv], -1, purple, 3)

        # Area
        area = cv2.contourArea(indiv)
        #print('area = ', area)

        # Perimeter
        perimeter = cv2.arcLength(indiv,True)
        #print('perimeter = ', perimeter)

        # Hull
        hull = cv2.convexHull(indiv)
        #print('hull', hull)
        #print('hull contour length = ', len(hull))
        #cv2.drawContours(imgContours, [hull], -1, orange, 5)
        #cv2.imshow('hull over green mask', imgContours)
        hull_area = cv2.contourArea(hull)
        #print('area of convex hull',hull_area)
        #print('solidity from convex hull', float(area)/hull_area)

        # straight bounding rectangle
        xb,yb,wb,hb = cv2.boundingRect(indiv)
        # store for passing to get_four
        bounding_rect = (xb,yb,wb,hb)
    
        #print('straight bounding rectangle = ', (xb,yb) ,wb,hb)
        cv2.rectangle(imgContours,(xb,yb),(xb+wb,yb+hb),green,2)
        #print('bounding rectangle aspect = ', float(wb)/float(hb))
        #print('bounding rectangle extend = ', float(area)/(float(wb)*float(hb)))

        # rotated rectangle
        rect = cv2.minAreaRect(indiv)
        #print('rotated rectangle = ',indiv)
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
        #print('minimum area rectangle aspect = ', float(wr)/hr)
        #print('minimum area rectangle extent = ', float(area)/(wr*hr))

        # send chosen contour to 4 point finder
        ROI_mask = binary_mask[yb:yb+hb, xb:xb+wb]
        intROMHeight, intROMWidth = ROI_mask.shape[:2]
        imgFindContourReturn, ROIcontours, hierarchy = cv2.findContours(ROI_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ROISortedContours = sorted(ROIcontours, key = cv2.contourArea, reverse = True)[:1]
        
        # send chosen contour to 4 point finder, get back found points
        try_get_four = get_four(bounding_rect,intROMWidth, intROMHeight, ROISortedContours[0])

        if try_get_four is None:
            pass

        else:
            rul, rbl, rbc, rbr, rur = try_get_four

            rulx, ruly = rul
            ulx = rulx + xb
            uly = ruly + yb

            rblx, rbly = rbl
            blx = rblx + xb
            bly = rbly + yb

            rbcx, rbcy = rbc
            bcx = rbcx + xb
            bcy = rbcy + yb

            rbrx, rbry = rbr
            brx = rbrx + xb
            bry = rbry + yb

            rurx, rury = rur
            urx = rurx + xb
            ury = rury + yb

            cv2.circle(imgContours, (ulx,uly), 12, green, -1)
            cv2.circle(imgContours, (blx,bly), 12, blue, -1)
            cv2.circle(imgContours, (bcx,bcy), 12, blue, -1)
            cv2.circle(imgContours, (brx,bry), 12, blue, -1)
            cv2.circle(imgContours, (urx,ury), 12, red, -1)

    # process Finish

    # update the FPS counter
    fps.update()

    # check to see if the frame should be displayed to our screen
    if display > 0:
        cv2.putText(imgContours, "frame number: " + str(int(fps._numFrames)), (40, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6 ,white)
        cv2.imshow('Frame', imgContours)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break


# stop the timer and display FPS information
fps.stop()
print('[INFO] elasped time: {:.2f}'.format(fps.elapsed()))
print('[INFO] approx. FPS: {:.2f}'.format(fps.fps()))

if not booWebCam:
    key = cv2.waitKey(0) & 0xFF

# do a bit of cleanup
if booWebCam:
    if booThreaded:
        vs.stop()
    else:
        stream.release()
else:
    pass

cv2.destroyAllWindows()