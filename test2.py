import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1 = cv.imread("CRW_4853_scale.jpg", cv.IMREAD_GRAYSCALE)
img2 = cv.imread("Capture.PNG", cv.IMREAD_GRAYSCALE)

# Initiate SIFT detector
sift = cv.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

# Apply ratio test
good = []
counter = 0

for m,n in matches:
    counter += 1
    if m.distance < 99*n.distance:
        good.append([n])
    else:
        print(m.distance , n.distance)
        
        
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good[:100],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3),plt.show()
print(counter)
print(len(good))
