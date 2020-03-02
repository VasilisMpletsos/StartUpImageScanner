import cv2
import os
import time

#Importing Key
key = cv2.imread('key.jpg',0)

#Capturing from Main Camera a frame in order to detect the key in it
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ret, frame = cap.read()
detect = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()

# Create SIFT Object
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(key,None)
kp2, des2 = sift.detectAndCompute(detect,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for match1,match2 in matches:
    if match1.distance < 0.75*match2.distance:
        good.append([match1])

# cv2.drawMatchesKnn expects list of lists as matches.
key2 = cv2.drawMatchesKnn(key,kp1,detect,kp2,good,None,flags=2)

#If you find more than 15 matching features then continue , else shutdown PC
if (len(good)>15):
    print("Found Match")
else:
    print("Match didn't found, Closing PC")
    os.system('cmd /k "%windir%/System32/shutdown.exe -s -t 00"')
