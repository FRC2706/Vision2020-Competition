import numpy as np
import cv2

avgFrames = [0 for i in range(0, 5)]
avgX = [0 for i in range(0, len(avgFrames))]
avgY = [0 for i in range(0, len(avgFrames))]
timer = [0 for i in range(0, 30)]
tupTime = (0, 0)
start = True
counter = 0
counter2 = 0
averageX = 0
averageY = 0
timeElapsed = 0
startX = 0

leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])

for cnt in avgFrames:
    if cnt == 0:
        cnt = leftmost

while start:
    del avgFrames[len(avgFrames) - 1]
    avgFrames.insert(0, leftmost)
    counter += 1
    if(counter == len(avgFrames) - 1):
        start = False

del avgFrames[len(avgFrames) - 1]
avgFrames.insert(0, leftmost)

print(avgFrames)

for x in range(0, len(avgX)):
    avgX[x] = (avgFrames[x])[0]

print(avgX)

for x in range(0, len(avgY)):
    avgY[x] = (avgFrames[x])[1]

print(avgY)

averageX = sum(avgX) / len(avgX)
averageY = sum(avgY) / len(avgY)
            
if(counter2 == 0):
    startX = averageX

counter2 = 1

print('Average Leftmost: ' + str(avgX))

while(startX < averageX):
    now = time.time()
    future = now + 30
    while time.time() < future:
        x = 0
        tupTime = (time.time(), averageX)
        timer[x] = tupTime
        x += 1
        timeElapsed = time.time()

print('The Time Elapsed: ' + str(timeElapsed))
print('Timer array ' + str(timer))
