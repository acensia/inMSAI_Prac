import cv2
import numpy as np

kalman = cv2.KalmanFilter(4,2)
kalman.measurementMatrix = np.array([[1,0,0,0],
                                     [0,1,0,0]], np.float32)

kalman.transitionMatrix = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)

kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]], np.float32) * 0.05


cap = cv2.VideoCapture("./data/slow_traffic_small.mp4")
print(cap)

ret, frame = cap.read()
print(ret, frame)