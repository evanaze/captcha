""" This script makes the full processed training data.
"""
import os
import numpy as np
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import cv2 as cv 
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.io import imsave

from .. import config
from ..features.preprocess import preprocess


class MakeData:
    "This class ingests the raw data and creates the full training set."
    def __init__(self):
        self.files = sorted(os.listdir(config.RAW_DIR))
        self.files = [f for f in self.files if not f.startswith('.')]
        self.n = len(self.files)


    def split_train_test(self):
        filenames, targets = [0]*self.n, np.empty(self.n, dtype=int)
        for idx, f in enumerate(self.files):
            filenames[idx] = f
            targets[idx] = int(f.split("_")[0])
        self.df_all = pd.DataFrame({"filename": filenames, "target": targets})
        # train test split
        X, y = self.df_all.filename, self.df_all.target - 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.RAND_STATE, shuffle=True)
        self.df_train = pd.DataFrame({"filename": X_train, "target": y_train}).reset_index(drop=True)
        self.df_test = pd.DataFrame({"filename": X_test, "target": y_test}).reset_index(drop=True)
        self.df_train.to_csv("data/train.csv", index=False)
        self.df_test.to_csv("data/test.csv", index=False)
        self.df_all.to_csv("data/all.csv", index=False)


    def make_synthetic(self):
        "Make 4 copies of the processed image"
        src = preprocess(self.image)
        f_split = self.f_out.split("."); f_split[1] = ".png"
        self.imgs = [0] * 4
        for i in range(4):
            angle = int(90 * i)
            rot = rotate(src, angle)
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
        "Turn the train split into processed data."
        m = len(self.df_train)
        df_proc = pd.DataFrame(
            columns=["filename", "target", "kfold"], 
            index=range(4*m)
        ).T
        # make the synthetic, processed data for each image
        for index, row in self.df_train.iterrows():
            print(f"Train: processing image {index} of {m}", end="\r")
            target = row["target"]
            f_name = row["filename"]
            # the name of the processed image
            self.f_out = "img" + str(index) + "_" + str(target) + ".png"
            # where to store the output image
            image_loc = os.path.join(config.RAW_DIR, f_name)
            # read the image
            self.image = cv.imread(image_loc)
            # generate synthetic data
            self.make_synthetic()
            # rotate the images
            for i, img_loc in enumerate(self.imgs):
                df_proc[4*index + i] = [img_loc, target, -1]
        print(); print("Done.")
        # add the fold numbers
        kf = KFold(n_splits=config.N_FOLDS)
        # we want the transpose
        df_proc = df_proc.T
        X_proc = df_proc["filename"]; y_proc = df_proc["target"]
        for fold, (train_idx, val_idx) in enumerate(kf.split(X=X_proc, y=y_proc)):
            df_proc.loc[val_idx, 'kfold'] = fold
        # save to csv
        df_proc.to_csv("data/train_proc.csv", index=False)


    def make_test_df(self):
        "makes the processed test data"
        m = len(self.df_test)
        df_proc = pd.DataFrame(
            columns=["filename", "target"], 
            index=range(4*m)
        ).T
        # make the synthetic, processed data for each image
        for index, row in self.df_test.iterrows():
            print(f"Test: processing image {index} of {m}", end="\r")
            target = row["target"]
            f_name = row["filename"]
            # the name of the processed image
            self.f_out = "img" + str(index) + "_" + str(target) + ".png"
            # where to store the output image
            image_loc = os.path.join(config.RAW_DIR, f_name)
            # read the image
            self.image = cv.imread(image_loc)
            # generate synthetic data
            self.make_synthetic()
            # rotate the images
            for i, img_loc in enumerate(self.imgs):
                df_proc[4*index + i] = [img_loc, target]
        print(); print("Done.")
        # save to csv
        df_proc.T.to_csv("data/test_proc.csv", index=False)


    def main(self):
        self.split_train_test()
        with ProcessPoolExecutor(max_workers=2) as ppe:
            futures = [ppe.submit(self.make_train_df()), 
            ppe.submit(self.make_test_df())]


if __name__ == "__main__":
    md = MakeData()
    md.main()
