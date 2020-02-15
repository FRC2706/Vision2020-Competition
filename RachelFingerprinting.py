if len(contours) >= 1:
        # Sort contours by area size (biggest to smallest)
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)[:17]
        contourImage = image.copy()

        cv2.drawContours(contourImage, cntsSorted, -1, (255, 0, 0), 5) #problem target contour was #10
        cv2.imshow('contourImage', contourImage)
        
        for (j, cnt) in enumerate(cntsSorted):

            print('j =', j) #which number contour is it

            # Calculate Contour area
            cntArea = cv2.contourArea(cnt)

            # rotated rectangle fingerprinting
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(contourImage,[box],0,(0, 0, 255),2) #drawing rotated rect
            cv2.imshow('contourImage', contourImage)#showing rotated rects
            (xr,yr),(wr,hr),ar = rect #x,y width, height, angle of rotation = rotated rect

            #to get rid of height and width switching
            if hr > wr: 
                ar = ar + 90
                wr, hr = [hr, wr]
            else:
                ar = ar + 180
            if ar == 180:
                ar = 0

            cntAspectRatio = float(wr)/hr
            print('cntAspectRatio = ', cntAspectRatio)
            minAextent = float(cntArea)/(wr*hr)
            print('minAextent = ', minAextent) 

            # Hull
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(cntArea)/hull_area
            print('solidity from convex hull', solidity)

            if (minAextent < 0.16 or minAextent > 0.26): continue
            if (cntAspectRatio < 2.0 or cntAspectRatio > 3.0): continue
            if (solidity < 0.22 or solidity > 0.30): continue
            #end fingerprinting