import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('download (2).png')

# Converting to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Applying threshold
ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

# Finding Contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Drawing Contours
img = cv2.drawContours(img, contours, -1, (0, 255, 0), 4)

plt.subplot(121), plt.imshow(gray)
plt.subplot(122), plt.imshow(img)
plt.show()

# Moments
cnt = contours[0]
M = cv2.moments(cnt)
print(M)

# Centroid
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])
print('Centroid = ', cx, ',', cy)

# Contour Area
area = cv2.contourArea(cnt)
print('Area = ', area)

# Contour perimeter
per = cv2.arcLength(cnt, True)
print('Cnt perimeter = ', per)

# BOUNDING SHAPES
image = cv2.imread('img_2.png')
gray2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret1, thresh2 = cv2.threshold(gray2, 100, 255, cv2.THRESH_BINARY)
cont, hier = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image = cv2.drawContours(image, cont, -1, (0, 255, 0), 4)
image2 = image.copy()
image3 = image.copy()

plt.subplot(121), plt.imshow(gray2)
plt.subplot(122), plt.imshow(image)
plt.show()

# Straight Bounding Rectangle
cnt1 = cont[0]
x, y, w, h = cv2.boundingRect(cnt1)
image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
cv2.imshow('Rectangle', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Rotated Bounding Rectangle
rect = cv2.minAreaRect(cnt1)
box = cv2.boxPoints(rect)
box = np.int0(box)
im = cv2.drawContours(image, [box], 0, (0, 0, 255), 3)
cv2.imshow('rotated rectangle', im)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Minimum Enclosing Circle
(x0, y0), radius = cv2.minEnclosingCircle(cnt1)
center = (int(x0), int(y0))
radius = int(radius)
im2 = cv2.circle(image2, center, radius, (100, 100, 10), 3)
cv2.imshow('circle', im2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Fitting an Ellipse
ellipse = cv2.fitEllipse(cnt1)
im3 = cv2.ellipse(image2, ellipse, (100, 50, 150), 3)
cv2.imshow('ellipse', im3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# CONTOURS PROPERTIES

# Aspect Ratio
# It is the ratio of width to height of bounding rectangle.
aspect_ratio = float(w) / h
print('Aspect Ratio = ', aspect_ratio)

# Extent
# Ratio of contour area to bounding rectangle area.
extent = float(area) / (w * h)
print('Extent = ', extent)

# Solidity
# Ratio of contour area to its convex hull area
hull = cv2.convexHull(cnt1)
hull_area = cv2.contourArea(hull)
solidity = float(area) / hull_area
print('Solidity = ', solidity)

# Equivalent Diameter
# Diameter of the circle whose area is same as the contour area.
equi_diameter = np.sqrt(4 * area / np.pi)
print('Equivalent Diameter = ', equi_diameter)

# Orientation
# The angle at which object is directed.
(x1, y1), (MA, ma), angle = cv2.fitEllipse(cnt1)
print('Orientation = ', angle)

# Extreme Points
leftmost = tuple(cnt1[cnt1[:, :, 0].argmin()][0])
rightmost = tuple(cnt1[cnt1[:, :, 0].argmax()][0])
topmost = tuple(cnt1[cnt1[:, :, 1].argmin()][0])
bottommost = tuple(cnt1[cnt1[:, :, 1].argmax()][0])
im4 = cv2.circle(image3, leftmost, 0, (0, 0, 255), 5)
im4 = cv2.circle(im4, rightmost, 0, (0, 0, 255), 5)
im4 = cv2.circle(im4, topmost, 0, (0, 0, 255), 5)
im4 = cv2.circle(im4, bottommost, 0, (0, 0, 255), 5)
cv2.imshow('points', im4)
cv2.waitKey(0)
cv2.destroyAllWindows()
