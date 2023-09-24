import random
import cv2
import albumentations as A
import matplotlib.pyplot as plt

img = cv2.imread("cat_dog.jpeg")

def visualize(img):
    cv2.imshow("", img)
    key = cv2.waitKey()
    return key
    
# transform = A.HorizontalFlip(p=0.5)
random.seed(7)

def img_mod(img):
    transform = A.Compose([
        A.CLAHE(),
        A.RandomRotate90(),
        A.Transpose(),
        A.ShiftScaleRotate(shift_limit=0.5, rotate_limit=45, p=.75),
        A.Blur(blur_limit=3),
        A.OpticalDistortion(),
        A.GridDistortion(),
        A.HueSaturationValue()
    ])

    while True:
        augmentated_img = transform(image=img)['image']
        key=visualize(augmentated_img)
        if(key == ord('q')):
            cv2.destroyAllWindows()
            break
    
    
img = cv2.imread("image02.jpeg")
def weather(img):
    transform = A.Compose([
        A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1),
        A.RandomSnow(brightness_coeff=2.5,snow_point_lower=0.3,snow_point_upper=0.5, p=1)
    ])
    while True:
        augmentated_img = transform(image=img)['image']
        key=visualize(augmentated_img)
        if(key == ord('q')):
            cv2.destroyAllWindows()
            break
        
weather(img)