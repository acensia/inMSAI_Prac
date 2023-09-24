import cv2
import glob
import matplotlib.pyplot as plt
import librosa
import librosa.display

import os
import numpy as np

from tqdm import tqdm
from PIL import Image


class myCustomTest(dataset):
    def __init__(self,img_path,transform):
        self.image_paths = glob.glob(os.path.join(img_path, "*", "*", "*.png"))
        self.transform = transform
        
        self.label_dict = {"Stealing_Courier" : 0}