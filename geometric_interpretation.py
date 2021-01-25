import cv2
import numpy as np
from matplotlib import pyplot as plt

# SCALING
img = cv2.imread('pic.png')
resize_image = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

cv2.imshow('resize', resize_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# TRANSLATION
rows, cols = img.shape[:2]
M = np.float32([[1, 0, 100], [0, 1, 50]])
translated_image = cv2.warpAffine(img, M, (cols, rows))

cv2.imshow('translation', translated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ROTATION
matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1)
rotated_image = cv2.warpAffine(img, matrix, (cols, rows))

cv2.imshow('rotation', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# AFFINE TRANSFORMATION
img1 = cv2.imread('affine.jpg')
row, col, ch = img1.shape

pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

mat = cv2.getAffineTransform(pts1, pts2)

affine_image = cv2.warpAffine(img1, mat, (col, row))

plt.subplot(121), plt.imshow(img1), plt.title('Input')
plt.subplot(122), plt.imshow(affine_image), plt.title('Affine_Image')
plt.show()
