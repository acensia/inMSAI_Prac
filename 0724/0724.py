import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

img_folder_path = "./dataset/images"
anno_folder_path = "./dataset/annotations/"

train_folder = "./dataset/train"
eval_folder = "./dataset/eval"
os.mkdir(train_folder)
os.mkdir(eval_folder)

csv_path = os.path.join(anno_folder_path, "annotations.csv")

anno_df = pd.read_csv(csv_path)

# print(anno_df)

img_names = anno_df['filename'].unique()
train_names, eval_names = train_test_split(img_names, test_size=0.2)

# print(train_names, eval_names)


train_annotations = pd.DataFrame(columns=anno_df.columns)
# print(train_annotations)

for img_name in train_names:
    # print("image_name value >> ", img_name)
    label = anno_df.loc[anno_df['filename'] == img_name, 'region_id']
    img_path = os.path.join(img_folder_path, img_name)
    new_img_path = os.path.join(train_folder, img_name)
    
    shutil.copy(img_path, new_img_path)
    
    annotation = anno_df.loc[anno_df['filename'] == img_name].copy()
    annotation['filename'] = f"{label}_{img_name}"
    # print(annotation)
    train_annotations = train_annotations._append(annotation)

# print(train_annotations)

train_annotations.to_csv(os.path.join(train_folder, 'annotations.csv'), index=False)

eval_annotations = pd.DataFrame(columns=anno_df.columns)
# print(train_annotations)

for img_name in eval_names:
    # print("image_name value >> ", img_name)
    label = anno_df.loc[anno_df['filename'] == img_name, 'region_id']
    
    img_path = os.path.join(img_folder_path, img_name)
    new_img_path = os.path.join(eval_folder, img_name)
    
    shutil.copy(img_path, new_img_path)
    
    annotation = anno_df.loc[anno_df['filename'] == img_name].copy()
    annotation['filename'] = f"{label}_{img_name}"
    # print(annotation)
    eval_annotations = eval_annotations._append(annotation)
    
    
eval_annotations.to_csv(os.path.join(eval_folder, 'annotations.csv'), index=False)