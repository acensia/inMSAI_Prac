from PIL import Image
import numpy as np

# List of your image file paths

for i in range(10):
    # Open image
    image_file = f"_최준혁2_{i}_before.jpg"
    img = Image.open(image_file)
    # img.save(f"_최준혁2_{i}_before.jpg")
    img = np.array(img)
    # Resize image to 10x10
    img_resized = Image.fromarray(img[160:480,160:480]).rotate(270).resize((30, 30))
    
    # Save resized image
    img_resized.save(f"최준혁2_{i}.jpg")
