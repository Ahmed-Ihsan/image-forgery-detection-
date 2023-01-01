import sys
import cv2
from test3_lip import Detect
import re
from datetime import datetime


file_name = 'CRW_4853tamp1.jpg'
image=cv2.imread(file_name)
if image is None:
	print(file_name)
	print('Enter Valid File Name/Path.')
	sys.exit(0)

eps =60
min_samples=2

flag=True
value = 50
if flag:
	try:
		value=int(value)
		if(value<0 or value> 500):
			print('Value not in range (0,500)........ using default value.')
		else:
			eps= value
	except ValueError:
		print('Value not integer........ using default value.')

flag2= True
value = 2

if flag2:
	try:
		value=int(value)
		if(value<0 or value> 50):
			print('Value not in range (0,50)........ using default value.')
		else:
			min_samples= value
	except ValueError:
		print('Value not integer........ using default value.')

print('Detecting Forgery with parameter value as\neps:{}\nmin_samples:{}'.format(eps,min_samples))

detect=Detect(image)

key_points,descriptors = detect.siftDetector()

forgery= detect.locateForgery(eps,min_samples)

if forgery is None:
	sys.exit(0)
	
cv2.imshow('Original image',image)
cv2.imshow('Forgery',forgery)
cv2.waitKey(0)
cv2.destroyAllWindows()


cv2.destroyAllWindows()
