import cv2
import numpy as np

img = cv2.imread('Dataset/yes/Y13.jpg')

#image to grayscale, and blurring it slightly
g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Thresholding the image, and performing a series of erosions + dilations to remove any regions of noise
thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)

#Finding the largest contour in the thresholded image
(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
con = cv2.drawContours(img, cnts, -1, (0, 0, 255), 2)
c = max(cnts, key=cv2.contourArea)

#Finding the extreme points
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])

#Cropping the image using extreme points (left, right, top, bottom)
new_img = g[extTop[1]:extBot[1], extLeft[0]:extRight[0]]            

cv2.imshow('contour', con)
cv2.imshow('Image', new_img)
cv2.waitKey(0)