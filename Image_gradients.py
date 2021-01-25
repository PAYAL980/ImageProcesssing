import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('SUDOKU.png')

# Laplacian Derivatives
laplacian = cv2.Laplacian(img, cv2.CV_64F)

lap_abs = np.absolute(laplacian)
lap_8u = np.uint8(lap_abs)

# Sobel Derivatives
sobelx = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)

plt.subplot(221), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(lap_8u), plt.title('Laplacian')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(sobelx), plt.title('SobelX')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(sobely), plt.title('SobelY')
plt.xticks([]), plt.yticks([])
plt.show()
