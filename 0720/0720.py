import cv2
import glob
img_lists = glob.glob("*.jpg")
img_list = []
gray_list = []
img_rots = []
gray_rots = []
for img in img_lists:
    img = cv2.imread(img)
    img = cv2.resize(img, (500, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    gray_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img_list.append(img)
    img_rots.append(img_rot)
    gray_list.append(gray)
    gray_rots.append(gray_rot)

sift = cv2.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(gray[3], None)
keypoints2, descriptors2 = sift.detectAndCompute(gray_rots[4], None)

matcher = cv2.BFMatcher()
matchers = matcher.match(descriptors, descriptors2)

matchers = sorted(matchers, key=lambda x: x.distance)

for match in matchers[:10]:
    print("Distance : ", match.distance)
    print("Keypoint 1 : (x=%d, y= %d)"%(int(keypoints[match.queryIdx].pt[0]),
                                        int(keypoints[match.queryIdx].pt[1])))
    print("Keypoint 2 : (x=%d, y= %d)"%(int(keypoints2[match.trainIdx].pt[0]),
                                        int(keypoints2[match.trainIdx].pt[1])))


matched_img = cv2.drawMatches(img_list[3], keypoints, img_rots[4], keypoints2,
                              matchers[:10], None,
                              flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("Matched IMage", matched_img)
cv2.waitKey(0)

# img_with_keypoints = cv2.drawKeypoints(img, keypoints, None)

# cv2.imshow("Img with Kp",img_with_keypoints)
# cv2.waitKey(0)
# cv2.destroyAllWindows()