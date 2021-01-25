import cv2
from matplotlib import pyplot as plt

# Reading the image and converting to grayscale
img = cv2.imread('pic.png')
image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# GLOBAL THRESHOLDING
retval, thresh1 = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY)
retval1, thresh2 = cv2.threshold(image, 110, 255, cv2.THRESH_BINARY_INV)
retval2, thresh3 = cv2.threshold(image, 110, 255, cv2.THRESH_TRUNC)
retval3, thresh4 = cv2.threshold(image, 110, 255, cv2.THRESH_TOZERO)
retval4, thresh5 = cv2.threshold(image, 110, 255, cv2.THRESH_TOZERO_INV)

titles = ['Original image', 'Thresh binary', 'Thresh Binary_Inv', 'Trunc', 'Tozero', 'Tozero_Inv']
images = [image, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# ADAPTIVE THRESHOLDING
adapt_thresh1 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
adapt_thresh2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

adapt_titles = ['original image', 'Thresh_binary', 'Adaptive_mean', 'Adaptive_gaussian']
adapt_images = [image, thresh1, adapt_thresh1, adapt_thresh2]

for i in range(4):
    plt.subplot(2, 2, i + 1), plt.imshow(adapt_images[i], 'gray')
    plt.title(adapt_titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

# Otsu's BINARIZATION
# Before Gaussian filtering
retval, otsu_thresh1 = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

# After Gaussian filtering
blur = cv2.GaussianBlur(image, (5, 5), 0)
retval5, otsu_thresh2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

otsu_titles = ['original image', 'Histogram', 'Global thresholding',
               'original image', 'Histogram', "Otsu's thresholding",
               'Gaussian filtered image', 'Histogram', "Otsu's thresholding"]
otsu_images = [image, 0, thresh1,
               image, 0, otsu_thresh1,
               blur, 0, otsu_thresh2]

for i in range(3):
    plt.subplot(3, 3, i * 3 + 1), plt.imshow(otsu_images[i * 3], 'gray')
    plt.title(otsu_titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 2), plt.hist(otsu_images[3 * i].ravel(), 256)
    plt.title(otsu_titles[3 * i + 2]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, 3 * i + 3), plt.imshow(otsu_images[3 * i + 2], 'gray')
    plt.title(otsu_titles[3 * i + 2]), plt.xticks([]), plt.yticks([])
plt.show()
