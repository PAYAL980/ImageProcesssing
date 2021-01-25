import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('000613.jpg')
img3 = cv2.imread('pexels-photo-614810.jpeg')
img2 = cv2.resize(img3, None, fx=1024/500, fy=1024/427)

# Gaussian pyramids for images
G = img1.copy()
gp1 = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gp1.append(G)

H = img2.copy()
gp2 = [H]
for i in range(6):
    H = cv2.pyrDown(H)
    gp2.append(H)

# Laplacian pyramids for images
lp1 = [gp1[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gp1[i])
    L1 = cv2.subtract(gp1[i - 1], GE)
    lp1.append(L1)

lp2 = [gp2[5]]
for i in range(5, 0, -1):
    HE = cv2.pyrUp(gp2[i])
    L2 = cv2.subtract(gp2[i - 1], HE)
    lp2.append(L2)

# Add left and right halves of images in each level
LS = []
for l1, l2 in zip(lp1, lp2):
    row, col, ch = l1.shape
    ls = np.hstack((l1[:, 0: int(col/2)], l2[:, int(col/2):]))
    LS.append(ls)

# Reconstruct
ls_ = LS[0]
for i in range(1, 6):
    ls_ = cv2.pyrUp(ls_)
    ls_ = cv2.add(ls_, LS[i])

# Direct combination
# real = np.hstack((img1[:, : col/2], img2[:, col/2:]))
real = img1 + img2

plt.subplot(221), plt.imshow(img1)
plt.subplot(222), plt.imshow(img2)
plt.subplot(223), plt.imshow(ls_)
plt.subplot(224), plt.imshow(real)
plt.show()
