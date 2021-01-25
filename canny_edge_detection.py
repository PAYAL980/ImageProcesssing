import cv2
from matplotlib import pyplot as plt

img = cv2.imread('pic.png')
canny = cv2.Canny(img, 50, 200)

plt.subplot(121), plt.imshow(img), plt.title('Origianl')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(canny), plt.title('Canny Image')
plt.xticks([]), plt.yticks([])
plt.show()


# CREATING TRACKBARS FOR THRESHOLD VALUES
def nothing(x):
    pass


# Creating Trackbars
cv2.namedWindow('image')
cv2.createTrackbar('lower threshold', 'image', 0, 255, nothing)
cv2.createTrackbar('Upper threshold', 'image', 0, 255, nothing)

while 1:
    cv2.imshow('image', canny)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    # Getting trackbars' positions
    lw = cv2.getTrackbarPos('lower threshold', 'image')
    up = cv2.getTrackbarPos('Upper threshold', 'image')

    canny2 = cv2.Canny(img, lw, up)
    canny[:] = canny2[:]
cv2.destroyAllWindows()
