import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('img_1.png')
kernel = np.ones((11, 11), np.uint8)

# EROSION
erosion = cv2.erode(img, kernel, iterations=3)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.subplot(122), plt.imshow(erosion), plt.title('Erosion')
plt.show()

# DILATION
dilation = cv2.dilate(img, kernel, iterations=2)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.subplot(122), plt.imshow(dilation), plt.title('Dilation')
plt.show()

# OPENING
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.subplot(122), plt.imshow(opening), plt.title('Opening')
plt.show()

# CLOSING
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.subplot(122), plt.imshow(closing), plt.title('Closing')
plt.show()

# MORPHOLOGICAL GRADIENT
morph_gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.subplot(122), plt.imshow(morph_gradient), plt.title('Morph_grad')
plt.show()

# TOP HAT
kernel1 = np.ones((29, 29), np.uint8)
top_hat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel1)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.subplot(122), plt.imshow(top_hat), plt.title('Top_hat')
plt.show()

# BLACK HAT
black_hat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel1)

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.subplot(122), plt.imshow(black_hat), plt.title('Black_hat')
plt.show()

# STRUCTURING KERNELS
rectangular_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
elliptical_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
cross_shaped_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
