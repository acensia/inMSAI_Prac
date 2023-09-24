import cv2
import glob
import matplotlib.pyplot as plt
import librosa
import librosa.display

import os
import numpy as np

from tqdm import tqdm
from PIL import Image

slash = "\\"

def create_folder(folder_name):
    submission_dir = "./image_extraction_data"
    final_dir = "./final_data"
    for dir_type in ["MelSpectogram", "STFT", "waveshow"]:
        os.make_dir(f"{submission_dir}{slash}{dir_type}{slash}{folder_name}")
        os.make_dir(f"{final_dir}{slash}{dir_type}{slash}{folder_name}")






if __name__ == "__main__":
    raw_data_path = "./raw_data"
    raw_data_path_list = glob.glob(os.path.join(raw_data_path, "*","*","*.wav"))
    
    
    for raw in tqdm(raw_data_path_list):
        if raw.split(slash)[-1] == "jazz.00054.wav":
            continue
        data, sr = librosa.load(raw)
        
        