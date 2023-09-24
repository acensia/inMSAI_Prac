import cv2
import os

cap=cv2.VideoCapture("./data/blooms-113004.mp4")

fps = 25

cnt = 0

if cap.isOpened():
    while True:
        ret, frame = cap.read()
        
        if ret:
            if (int(cap.get(1)) % fps == 0):
                os.makedirs("./frame_img_data",exist_ok=True)
                cv2.imwrite(
                    f"./fame_img_data/img_{str(cnt).zfill(4)}.png",frame
                )
                
        else:
            break

cap.release()
cv2.destroyAllWindows()