import cv2
import numpy as np

# To find hsv range of a color
red = np.uint8([[[0, 0, 255]]])
res = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)
print(res)

# To find flags starting with COLOR_
flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
print(flags)

img1 = cv2.imread('photo-1557296387-5358ad7997bb.jfif')
img2 = cv2.imread('images (2).jfif')
print(img1.shape)
print(img2.shape)
img3 = cv2.resize(img2, None, fx= 1000/183, fy= 1271/275)
print(img3.shape)
#cv2.imshow('img', img3)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
