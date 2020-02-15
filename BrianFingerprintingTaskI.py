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