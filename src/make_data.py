import os
import numpy as np
import warnings

import pandas as pd
import cv2 as cv 
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
from skimage.transform import rotate

from skimage.io import imsave
from . import config
from .preprocess import preprocess

""" This script makes the full processed training data.

    The raw data must be placed in the input/raw directory.
"""

class MakeData:
    def __init__(self):
        self.files = sorted(os.listdir(config.RAW_DIR))
        self.files = [f for f in self.files if not f.startswith('.')]
        self.n = len(self.files)
        self.df_all = pd.DataFrame(columns=["filename", "target"], index=np.empty(self.n))

    def split_train_test(self):
        for idx, f in enumerate(self.files):
            self.df_all.loc[idx, ["filename", "target"]] = (f, int(f.split("_")[0]))
        # train test split
        X, y = self.df_all.filename, self.df_all.target - 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RAND_STATE, shuffle=True)
        self.df_train = pd.DataFrame({"filename": X_train, "target": y_train}).reset_index(drop=True)
        df_test = pd.DataFrame({"filename": X_test, "target": y_test})
        self.df_train.to_csv("input/train.csv", index=False)
        df_test.to_csv("input/test.csv", index=False)
        self.df_all.to_csv("input/all.csv", index=False)

    def make_synthetic(self):
        "Make 4 copies of the processed image"
        thresh = preprocess(self.image)
        f_split = self.f_out.split("."); f_split[1] = ".png"
        self.imgs = [0] * 4
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
            img_loc = config.PROC_DIR+'/'+img_name
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imsave(img_loc, img)
            self.imgs[i] = img_name

    def make_train_df(self):
        df_proc = pd.DataFrame(columns=["filename", "target", "kfold"], index=range(4*self.n)).T
        count = 0
        print("Creating synthetic data")
        for index, row in self.df_train.iterrows():
            target = row["target"]
            f_name = row["filename"]
            self.f_out = "img" + str(count) + "_" + str(target) + ".png"
            image_loc = os.path.join(config.DATA_DIR, f_name)
            image = cv.imread(image_loc)
            self.make_synthetic()
            for i, img_loc in enumerate(self.imgs):
                df_proc[4*count + i] = [img_loc, target]
            count += 1
        kf = KFold(n_splits=config.N_FOLDS)
        X_proc = df_proc["filename"]; y_proc = df_proc["target"]
        for fold, (train_idx, val_idx) in enumerate(kf.split(X=X_proc, y=y_proc)):
            df_proc.loc[val_idx, 'kfold'] = fold
        df_proc.T.to_csv("input/train_proc.csv", index=False)

    def main(self):
        self.split_train_test()
        self.make_train_df()


if __name__ == "__main__":
    md = MakeData()
    md.main()
