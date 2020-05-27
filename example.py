import numpy as np
import cv2

img = cv2.imread('Data/Ground_Truth/1.png', 0)/255.0
# img = cv2.imread('res.jpg', 0)
cv2.imshow('img1', img)

print(img.tolist())


a = 122.5
img[img > a] = 255
img[img <= a] = 0

cv2.imshow('img', img)
cv2.waitKey(0)

