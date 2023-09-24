import cv2

# init rectangle for mean shift tracking
track_window = None #temp for object loc data
roi_hist = None #temp for histogram
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

cap = cv2.VideoCapture("./data/slow_traffic_small.mp4")

ret, frame = cap.read()
# print(ret, frame)


x, y, w, h = cv2.selectROI("selectROI", frame, False, False)

#calculate init histogram of tracked obj
roi = frame[y:y+h, x:x+w]

# cv2.imshow("roi test", roi)
# cv2.waitKey(0)

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 100])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# set init window for tracked obj
track_window = (x, y, w, h)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv],[0],roi_hist, [0, 180], 1)
    
    _, track_window = cv2.meanShift(dst, track_window, term_crit)
    
    x, y, w, h = track_window
    print("추적 결과 좌표", x, y, w, h)
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)
    cv2.imshow("MeanShift Tracking",frame)
    
    if cv2.waitKey(30) & 0xFF==ord('q'):
        exit()

cap.release()
cv2.destroyAllWindows()