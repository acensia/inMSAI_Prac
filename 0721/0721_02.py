import cv2
import numpy as np

# load image

image = cv2.imread("./fass1.jpg")

grid_size = (8, 8)

def create_grid(img, grid_size= grid_size):
    h, w = image.shape[:2]
    grid_width , grid_height = grid_size
    grid_image = np.copy(image)
    for x in range(0, w, grid_width):
        cv2.line(grid_image, (x,0), (x,h), (0,255,0), 1)
    for y in range(0, h, grid_height):
        cv2.line(grid_image, (0,y), (w,y), (0,255,0), 1)
        
        
    return grid_image


gridimg = create_grid(image)

cv2.imshow("og",image)
cv2.imshow("grid", gridimg)

cv2.waitKey(0)
cv2.destroyAllWindows()