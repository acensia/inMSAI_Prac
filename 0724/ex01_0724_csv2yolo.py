import os
import pandas as pd
import json
from PIL import Image
from tqdm import tqdm


train_folder_path = "./train"
val_folder_path = "./eval"

train_csv_file_path = os.path.join(train_folder_path, 'annotations.csv')
val_csv_file_path = os.path.join(val_folder_path, 'annotations.csv')

train_anno_df = pd.read_csv(train_csv_file_path)
val_anno_df = pd.read_csv(val_csv_file_path)

def reszie_and_scale_bbox(img, bbox, target_size):
    img_w, img_h = img.size
    
    img = img.resize(target_size, Image.LANCZOS)
    resize_img_w, resize_img_h = img.size
    
    x, y, w, h = bbox
    x_scale = target_size[0] / img_w
    y_scale = target_size[1] / img_h
    
    x_center = (x + w/2)*x_scale
    y_center = (y + h/2)*y_scale
    
    scaled_w = w*x_scale
    scaled_h = h*y_scale
    scaled_bbox = (x_center, y_center, scaled_w, scaled_h)
    
    return img, scaled_bbox

#YOLO
def convert_to_yolo_format(annotation_df, og_image_folder, output_folder, target_size):
    for idx, row in tqdm(annotation_df.iterrows()):
        img_name = row['filename']
        label = row['region_id']
        
        img_path = os.path.join(og_image_folder, img_name)
        new_img_path = os.path.join(output_folder, "images",img_name)
        
        shape_attributes = json.loads(row['region_shape_attributes'])
        print("shape_attributes", shape_attributes)
        
        x = shape_attributes['x']
        y = shape_attributes['y']
        width = shape_attributes['width']
        height = shape_attributes['height']
        print(x, y, width, height)       
        
        img = Image.open(img_path)
        
        # img, scaled_boxs = re
        
train_yolo_folder_path = "./yolo_dataset/train"
val_yolo_folder_path = "./yolo_dataset/val"

os.makedirs(os.path.join(train_yolo_folder_path, "images"), exist_ok=True)
os.makedirs(os.path.join(train_yolo_folder_path, "labels"), exist_ok=True)

os.makedirs(os.path.join(val_yolo_folder_path, "images"), exist_ok=True)
os.makedirs(os.path.join(val_yolo_folder_path, "labels"), exist_ok=True)
    
    
"""
yolo_dataset
    train
        images
            aaa.png
        labels
            aaa.txt
    val
"""