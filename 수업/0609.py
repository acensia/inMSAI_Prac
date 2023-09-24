import cv2


cap=cv2.VideoCapture("./data/blooms-113004.mp4")

w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

print(f"OG w & h : {w}x{h}")
print(f"fps:{fps}")
print(f"frame count : {frame_count}")

if cap.isOpened():
    print("check init")
    while True:
        ret, frame = cap.read() #read next fr
        if not ret: #fail to read fr
            break
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow("video test", frame)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            exit()

else:
    print("failed to init obj")

cap.release()
cv2.destroyAllWindows()