import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("image02.jpeg")

# cv2.imshow("", img)
# cv2.waitKey()
# cv2.destroyAllWindows()

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, c = img.shape

def img_rot(img, h, w):
    angle = 30
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)


    rotated_img = cv2.warpAffine(img, M, (w, h))

    plt.imshow(img)
    plt.show()

    plt.imshow(rotated_img)
    plt.show()

def img_zoom(img, h, w):
    zoom_scale = 4
    zoomed_img = cv2.resize(img, (w*zoom_scale, h*zoom_scale), interpolation=cv2.INTER_CUBIC)

    plt.imshow(img)
    plt.show()
    plt.imshow(zoomed_img)
    plt.show()
    
def img_shift(img, h, w):
    pass

kernel_size = 15
kernel_direction = np.zeros((kernel_size, kernel_size))
kernel_direction[int((kernel_size)//2), :] = np.ones(kernel_size)
kernel_direction /= kernel_size
kernel_matrix = cv2.getRotationMatrix2D((kernel_size/2,kernel_size/2), 45, 1)
kernel = np.hstack((kernel_matrix[:, :2], [[0],[0]]))
# rotate mat 값만 가져옴
# 병진이동은 0,0으로 바꾸는?

motion_blur_img = cv2.filter2D(img, -1, kernel)

plt.imshow(motion_blur_img)
plt.show()

