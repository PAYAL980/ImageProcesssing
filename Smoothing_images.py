import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.util import random_noise

img2 = cv2.imread('images.jfif')
img = cv2.imread('opencvlogo.png')

# Image Filtering
kernel = np.ones((5, 5), np.float32) / 25

filtered_image = cv2.filter2D(img, -1, kernel)

plt.subplot(121), plt.imshow(img), plt.title('Original image')
plt.subplot(122), plt.imshow(filtered_image), plt.title('Filtered image')
plt.show()

# Image Blurring (AVERAGING)
blur = cv2.blur(img, (5, 5))

plt.subplot(121), plt.imshow(img), plt.title('Original image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur), plt.title('Blurred image')
plt.xticks([]), plt.yticks([])
plt.show()

# GAUSSIAN FILTERING
gaussian_blur = cv2.GaussianBlur(img2, (5, 5), 0)
gaussian_blur2 = cv2.GaussianBlur(img2, (7, 7), 0)

plt.subplot(131), plt.imshow(img2), plt.title('Original image')
plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(gaussian_blur), plt.title('Gaussian_Blurred image')
plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(gaussian_blur2), plt.title('Gaussian_Blurred2 image')
plt.xticks([]), plt.yticks([])
plt.show()

# MEDIAN FILTERING
median_blur = cv2.medianBlur(img2, 3)

plt.subplot(121), plt.imshow(img2), plt.title('Original image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(median_blur), plt.title('Median_Blurred image')
plt.xticks([]), plt.yticks([])
plt.show()

# BILATERAL FILTERING
bilateral_blur = cv2.bilateralFilter(img2, 9, 75, 75)

plt.subplot(121), plt.imshow(img2), plt.title('Original image')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(bilateral_blur), plt.title('Bilateral_Blurred image')
plt.xticks([]), plt.yticks([])
plt.show()

# Adding Gaussian noise and Salt and Pepper noise to image
img_gauss = random_noise(img, mode='gaussian', var=1.0)
img_gauss = np.array(255 * img_gauss, dtype='uint8')
noise_image_snp = random_noise(img, mode='s&p', amount=0.3)
noise_image_snp = np.array(255 * noise_image_snp, dtype='uint8')

blur1 = cv2.GaussianBlur(img_gauss, (5, 5), 0)
blur2 = cv2.GaussianBlur(noise_image_snp, (5, 5), 0)

plt.subplot(221), plt.imshow(img_gauss), plt.title('Gaussian noise')
plt.subplot(222), plt.imshow(noise_image_snp), plt.title('Salt and pepper noise')
plt.subplot(223), plt.imshow(blur1), plt.title('Gaussian bl')
plt.subplot(224), plt.imshow(blur2), plt.title('SnP noise')
plt.show()
