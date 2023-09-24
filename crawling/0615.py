import json
import os
import cv2
import glob
import numpy as np


json_dir = "./anno"
json_paths = glob.glob(os.path.join(json_dir, "*.json"))

label_dict = {"수각류" : 0}

for json_path in json_paths:
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
        
    imgs_info = json_data["images"]
    annos_info = json_data["annotations"]
    
    filename = imgs_info["filename"]
    img_id = imgs_info["id"]
    img_width = imgs_info["width"]
    img_height = imgs_info["height"]
    
    new_width = 1024
    new_height = 768
    
    for anno_info in annos_info:
        if img_id == anno_info["image_id"]:
            img_path = os.path.join("./images/", filename)
            img = cv2.imread(img_path)
            
            scale_x = new_width / img.shape[1]
            scale_y = new_height / img.shape[0]
            
            resized_img = cv2.resize(img, (new_width, new_height))
            
            category_name = anno_info["category_name"]
            polygons = anno_info["polygon"]
            
            points = []
            
            for polygon_info in polygons:
                x = polygon_info['x']
                y = polygon_info['y']
                
                resized_x = int(x*scale_x)
                resized_y = int(y*scale_y)
 
                points.append((resized_x, resized_y))
                cv2.polylines(resized_img,
                          [
                              np.array(points, np.int32).reshape((-1, 1, 2))
                          ],
                          True,
                          color=(0, 255, 0),
                          thickness=2)
            
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            x_min = min(x_coords)
            y_min = min(y_coords)
            x_max = max(x_coords)
            y_max = max(y_coords)
            
            cv2.rectangle(resized_img, (x_min, y_min), (x_max, y_max), (0, 0, 255))
            
            # print(f"")
            # cv2.imshow("",resized_img)
            # key = cv2.waitKey()
            # if key==ord('q'):
            #     exit()
            
            
            center_x = ((x_max + x_min)/(2*new_width))
            center_y = ((y_max + y_min)/(2*new_height))
            
            yolo_w = (x_max-x_min)/new_width
            yolo_h = (y_max-y_min)/new_height
            
            img_name_temp=filename.replace(".jpg","")
            
            temp_str = "C_TP_15_00007351.jpg"
            label_num = label_dict[category_name]
        os.makedirs("./yolo_label_data", exist_ok=True)
        with open(f"./yolo_label_data/{img_name_temp}.txt", "a") as f:
            f.write(f"{label_num} {center_x} {center_y} {yolo_w} {yolo_h}")
