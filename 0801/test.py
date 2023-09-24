import os
import glob

model = YOLO("./car_best.pt")
data_path = "./car_data"
data_path_List = glob.glob(os.path.join(data_path,"*.png"))

for path in data_path_list:
    img = cv2.imread(path)
    
    names = model.names
    
    res = model.predict(path, save=False, imgsz=640, conf=0.7)
    boxes = results[0].boxes
    print(boxes)
    
    exit()