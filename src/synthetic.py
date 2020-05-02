import os
import numpy as np
import pandas as pd
import cv2 as cv 
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.io import imsave
from torchvision import transforms
from . import config


def make_synthetic(image, f_out):
    laplacian = cv.Laplacian(image, cv.CV_64F)
    squares_channel = cv.split(laplacian)[0].astype(np.uint8)
    thresh = cv.threshold(squares_channel, 60, 255, cv.THRESH_BINARY)[1]
    f_split = f_out.split("."); f_split[1] = ".png"
    for i in range(4):
        angle = int(90 * i)
        rot = rotate(thresh, angle)
        img = cv.normalize(
            src=rot, 
            dst=None, 
            alpha=0, 
            beta=255, 
            norm_type=cv.NORM_MINMAX, 
            dtype=cv.CV_8U
        )
        img_name = f_split[0] + "_" + str(angle) + f_split[1]
        imsave("input/captcha_de-noised/"+img_name, img)
    

def main():
    df_all = pd.read_csv("input/all.csv")
    count = 0
    for index, row in df_all.iterrows():
        target = row["target"]
        f_name = row["filename"]
        f_out = "img" + str(count) + "_" + str(target) + ".png"
        image_loc = os.path.join(config.DATA_DIR, f_name)
        image = cv.imread(image_loc)
        make_synthetic(image, f_out)
        count += 1
        

if __name__ == "__main__":
    main()