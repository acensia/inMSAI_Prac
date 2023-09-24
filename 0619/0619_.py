import numpy as np
import imgaug.augmenters as iaa

import cv2
import matplotlib.pyplot as plt

file_path = "./sample_data_01/train/snow/0830.jpg"
file_path = "sample_data_01\\train\\snow\\0830.jpg"
img = cv2.imread(file_path)

imgs = [img, img, img, img]


def first():
    rotate = iaa.Affine(rotate=(-25, 25))
    imgs_aug = rotate(images=imgs)


    plt.figure(figsize=(12, 12))
    plt.imshow(np.hstack(imgs_aug))
    plt.show()

seq = iaa.OneOf([
    iaa.Grayscale(alpha=(0.0, 1.0)),
    iaa.AddToSaturation((-50 ,50))
])
imgs_aug04 = seq(images=imgs)

plt.figure(figsize=(12, 12))
plt.imshow(np.hstack(imgs_aug04))
plt.show()




# cv2.imshow("", img)
# cv2.waitKey()
# cv2.destroyAllWindows()