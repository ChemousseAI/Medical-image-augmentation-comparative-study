
import os
import glob
import cv2
import pandas as pd

def load_images(folder):
    images = []
    image_names = sorted(
        [f for f in os.listdir(folder)
         if f.endswith(('.jpg','.jpeg','.png','.bmp','.gif'))]
    )
    for image_name in image_names:
        image_path = os.path.join(folder, image_name)
        img = cv2.imread(image_path)
        images.append(img)
    return images

def load_labels(csv_path):
    labels = pd.read_csv(csv_path)
    return labels.iloc[:, [1]]

def load_idrid_dataset(base_path):
    paths = glob.glob(base_path + "/*")
    return paths
