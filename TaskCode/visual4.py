#

import numpy as np
import cv2

black = (0, 0, 0)
purple = (165, 0, 120)
green = (0, 255, 0)
red = (0, 0, 255)
blue = (255, 0, 0)
booWrite = True

def get_four(width, height, contour):
    """
    """

    # ignore small contours in case caller does not have minimum filter
    if width or height > 10:

        #calculate middle, thickness and radius for circles
        middle = int(width/2)
        thickness = int(height/10)
        divider = int(height/10) #was 7 before fifth

        # make mask from blank image and contour
        binary_mask = np.zeros([height,width,1],dtype=np.uint8)
        cv2.drawContours(binary_mask, [contour], -1, 255, cv2.FILLED)
        if booWrite: cv2.imwrite('./01-binary_mask.jpg', binary_mask)

        #find leftmost and rightmost
        leftmost = tuple(contour[contour[:,:,0].argmin()][0])
        rightmost = tuple(contour[contour[:,:,0].argmax()][0])
        
        # make negative of mask and close with line, divide in half
        negative_mask = cv2.bitwise_not(binary_mask)
        if booWrite: cv2.imwrite('02-negative_mask.jpg', negative_mask)
        modified_mask = negative_mask.copy()
        cv2.line(modified_mask, leftmost, rightmost, black, thickness, cv2.LINE_AA)
        cv2.line(modified_mask, (middle,0), (middle,height), black, divider, cv2.LINE_AA)
        if booWrite: cv2.imwrite('03-added_lines.jpg', modified_mask)

        #if booWrite: img123visual4 = np.hstack([binary_mask, negative_mask, modified_mask])

        # make hull and image of hull for bitwise_and
        hull = cv2.convexHull(contour)
        hull_mask = np.zeros([height,width,1],dtype=np.uint8)    
        cv2.drawContours(hull_mask, [hull], -1, 255, cv2.FILLED)
        if booWrite: cv2.imwrite('04-binary_hull.jpg', hull_mask)

        # bitwise_and the modified negative_mask and hull_mask
        bitwise_bottoms = cv2.bitwise_and(modified_mask,hull_mask)
        if booWrite: cv2.imwrite('05-bitwise_bottoms.jpg',bitwise_bottoms)

        # make contours of bitwise_and and keep largest two
        imgFindContourReturn, bitwiseContours, hierarchy = cv2.findContours(bitwise_bottoms, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sortedContours = sorted(bitwiseContours, key = cv2.contourArea, reverse = True)[:2]

        # sort contours so we know left vs right
        # https://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
        boundingBoxes = [cv2.boundingRect(c) for c in sortedContours]
        (sortedContours, boundingBoxes) = zip(*sorted(zip(sortedContours, boundingBoxes),
            key=lambda b:b[1][0], reverse=True))

        # approximate four corners of both
        leftHalf = sortedContours[0]
        epsilon = 0.035*cv2.arcLength(leftHalf,True)
        leftApprox = cv2.approxPolyDP(leftHalf,epsilon,True)
        rightHalf = sortedContours[1]
        epsilon = 0.035*cv2.arcLength(rightHalf,True)
        rightApprox = cv2.approxPolyDP(rightHalf,epsilon,True)

        # draw approximated contour
        two_bottoms = cv2.cvtColor(bitwise_bottoms,cv2.COLOR_GRAY2RGB)
        cv2.drawContours(two_bottoms, [leftApprox], -1, purple, 5)
        cv2.drawContours(two_bottoms, [rightApprox], -1, purple, 5)
        if booWrite: cv2.imwrite('06-largest_two.jpg', two_bottoms)

        #if booWrite: img456visual4 = np.hstack([hull_mask, bitwise_bottoms, two_bottoms])

        # draw points of approx on top of contours
        approx_points = cv2.cvtColor(bitwise_bottoms,cv2.COLOR_GRAY2RGB)
        cv2.drawContours(approx_points, leftApprox, -1, green, thickness)
        cv2.drawContours(approx_points, rightApprox, -1, red, thickness)
        if booWrite: cv2.imwrite('07-approx_points.jpg',approx_points)

        # left contour, sort by Y axis first to get lower points, then sort lower points by X
        # https://stackoverflow.com/questions/41284421/sort-a-numpy-array-according-to-2nd-column-only-if-values-in-1st-column-are-same
        [leftApprox] = leftApprox[np.lexsort(leftApprox[:,::-1].T)] # sort y axis then x axis
        leftApprox = leftApprox[-2:] # filter to lowest pair of coordintes, x descending
        [leftApprox] = leftApprox[np.lexsort(leftApprox[:,::1].T)] # sort x axis (1st is x) ascending

        # right contour, sort by Y axis first to get lower points, then sort lower points by X
        #https://stackoverflow.com/questions/41284421/sort-a-numpy-array-according-to-2nd-column-only-if-values-in-1st-column-are-same
        [rightApprox] = rightApprox[np.lexsort(rightApprox[:,::-1].T)] # sort y axis then x axis
        rightApprox = rightApprox[-2:] # filter to lowest pair of coordintes, x descending
        [rightApprox] = rightApprox[np.lexsort(rightApprox[:,::-1].T)] # sort x axis (1st is x) ascending

        # potential fifth point, average of inner points, from split, maybe
        [[lx5, ly5]] = leftApprox[1]
        [[rx5, ry5]] = rightApprox[0]
        fifth = (int((lx5+rx5)/2),int((ly5+ry5)/2))

        # first left is leftmost lower x, second right is rightmost lower x 
        [arrayLeft] = leftApprox[0]
        bottomcenter = fifth
        [arrayRight] = rightApprox[1]

        # extract out desired bottom corners
        [blx, bly] = arrayLeft
        [brx, bry] = arrayRight
        bottomleft = (blx, bly)
        bottomright = (brx, bry)

        # draw found corners
        color_bottoms = cv2.cvtColor(bitwise_bottoms, cv2.COLOR_GRAY2RGB)
        cv2.circle(color_bottoms, bottomleft, thickness, green, -1)
        cv2.circle(color_bottoms, bottomright, thickness, red, -1)
        cv2.circle(color_bottoms, bottomcenter, thickness, blue, -1)
        if booWrite: cv2.imwrite('08-bottom_points.jpg',color_bottoms)

        # draw four corners and center
        four_corners = cv2.cvtColor(bitwise_bottoms, cv2.COLOR_GRAY2RGB)
        cv2.circle(four_corners, leftmost, int(thickness/2), green, -1)
        cv2.circle(four_corners, rightmost, int(thickness/2), red, -1)
        cv2.circle(four_corners, bottomleft, int(thickness/2), blue, -1)
        cv2.circle(four_corners, bottomcenter, int(thickness/2), blue, -1)
        cv2.circle(four_corners, bottomright, int(thickness/2), blue, -1)
        if booWrite: cv2.imwrite('09-four_points.jpg',four_corners)

        if booWrite: img789visual4 = np.hstack([approx_points, color_bottoms, four_corners])

        if booWrite: 
            imgTraceVisual4 = np.vstack([img789visual4])
            cv2.imshow('Visual4 Trace Steps', imgTraceVisual4)
            #cv2.imwrite('10-Vision4Steps',imgTraceVisual4)
            cv2.moveWindow('Visual4 Trace Steps',350,760)

        return [leftmost,bottomleft,bottomcenter,bottomright,rightmost]

    else:
        return None
