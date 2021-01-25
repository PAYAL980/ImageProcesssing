import cv2
import numpy as np

capture = cv2.VideoCapture(0)
while 1:
    # Take each frame
    _, frame = capture.read()

    # Convert from BGR to HSV color-space
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range of blue and red color in hsv
    blue_lower = np.array([110, 50, 50])
    blue_upper = np.array([130, 255, 255])
    red_lower = np.array([0, 50, 50])
    red_upper = np.array([10, 255, 255])

    # Threshold HSV image to get blue and red color
    mask1 = cv2.inRange(frame_hsv, blue_lower, blue_upper)
    mask2 = cv2.inRange(frame_hsv, red_lower, red_upper)
    mask = mask1 + mask2

    # Bitwise AND mask and original frame
    result = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('result', result)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
