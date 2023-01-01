import cv2
import numpy as np

img = cv2.imread("CRW_4853_scale.jpg", cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
##surf = cv2.xfeatures2D.SURF_create()
orb = cv2.ORB_create(nfeatures=1500)

keypoints_sift, descriptors = sift.detectAndCompute(img, None)
##keypoints_surf, descriptors = surf.detectAndCompute(img, None)
keypoints_orb, descriptors = orb.detectAndCompute(img, None)

img1 = cv2.drawKeypoints(img, keypoints_sift, None)
img2 = cv2.drawKeypoints(img, keypoints_sift, None)

cv2.imshow("Image sift", img1)
cv2.imshow("Image orb", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
