import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

image_path = "./fass1.jpg"
img = Image.open(image_path)

transform = transforms.Compose([
    transforms.ToTensor()
])
img_tensor = transform(img).float()

grid_size = 16
height, width = img_tensor.shape[1], img_tensor.shape[2]

grid_width = width // grid_size
grid_height = height // grid_size

grids = []

for i in range(grid_size):
    for j in range(grid_size):
        x_min = j* grid_width
        y_min = i*grid_height
        x_max = (j+1)*grid_width
        y_max = y_min + grid_height
        
        grid = img_tensor[:, y_min:y_max, x_min:x_max]
        grids.append(grid)


fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))

for i in range(grid_size):
    for j in range(grid_size):
        axs[i,j].imshow(grids[i*grid_size + j].permute(1,2,0))
        axs[i,j].axis('off')
        
plt.show()