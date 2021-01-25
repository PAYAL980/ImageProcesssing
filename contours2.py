import cv2

img = cv2.imread('img_3.png')
image1 = img.copy()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)

# CONVEXITY DEFECTS
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]

hull = cv2.convexHull(cnt, returnPoints=False)
defects = cv2.convexityDefects(cnt, hull)

for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    start = tuple(cnt[s][0])
    end = tuple(cnt[e][0])
    far = tuple(cnt[f][0])
    cv2.line(img, start, end, (0, 255, 0), 3)
    cv2.circle(img, far, 5, (0, 0, 255), -1)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# POINT POLYGON TEST
# Finds the shortest distance of a given point from a contour.
point = (50, 50)
dist = cv2.pointPolygonTest(cnt, point, True)
print('Distance of contour from given point = ', dist)

# MATCH SHAPES
image2 = cv2.imread('img_2.png')
img_gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
ret1, thresh2 = cv2.threshold(img_gray2, 100, 255, cv2.THRESH_BINARY)

contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt2 = contours2[0]

ret0 = cv2.matchShapes(cnt, cnt2, 1, 0.0)
print(ret0)

# CONTOUR HIERARCHY
# Contour Retrieval Modes

# RETR_LIST
#  Parents and kids are equal under this rule,
#  and they are just contours. ie they all belongs to same hierarchy level.

# RETR_EXTERNAL
#  We can say, under this law,
#  Only the eldest in every family is taken care of.
#  It doesn’t care about other members of the family :).

# RETR_CCOMP
# This flag retrieves all the contours and arranges them to a 2-level hierarchy.
# ie external contours of the object (ie its boundary) are placed in hierarchy-1.
# And the contours of holes inside object (if any) is placed in hierarchy-2.
# If any object inside it, its contour is placed again in hierarchy-1 only.
# And its hole in hierarchy-2 and so on.

# RETR_TREE
# It retrieves all the contours and creates a full family hierarchy list.
# It even tells, who is the grandpa, father, son, grandson and even beyond... :).
