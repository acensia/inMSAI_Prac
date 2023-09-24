import cv2
import numpy as np

orb = cv2.ORB_create()

ob_img = cv2.imread("./fas2.jpg")
ob_img = cv2.resize(ob_img, (400, 400))

ob_gray = cv2.cvtColor(ob_img, cv2.COLOR_BGR2GRAY)

ob_keypoint, ob_descriptor = orb.detectAndCompute(ob_gray, None)

target_img = ob_img.copy()
target_gray = ob_gray.copy()

target_keypoint, target_descriptors = orb.detectAndCompute(target_img, target_gray)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matchers = matcher.match(ob_descriptor, target_descriptors)

matchers = sorted(matchers, key= lambda x:x.distance)

if len(matchers) > 10 :
    obj_found = True
    
    obj_points = [ob_keypoint[m.queryIdx].pt for m in matchers]
    tar_points = [target_keypoint[m.trainIdx].pt for m in matchers]
    
    obj_x, obj_y, obj_w, obj_h = cv2.boundingRect(np.float32(obj_points))
    target_x, target_y, target_w, target_h = cv2.boundingRect(np.float32(tar_points))
    
    cv2.rectangle(target_img, (obj_x, obj_y),
                  (obj_x+obj_w, obj_y+obj_h), (0, 255, 0), 2)
    
cv2.imshow("bb", target_img)
cv2.waitKey(0)
cv2.destroyAllWindows()