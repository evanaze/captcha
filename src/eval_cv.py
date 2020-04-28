import os
import pandas as pd
from sklearn.metrics import f1_score, precision_score, accuracy_score
import cv2 as cv 
from .n_squares import n_squares
from . import config


def main():
    df_all = pd.read_csv("input/all.csv")
    y_pred, y_true = [], []
    for index, row in df_all.iterrows():
        true = row["target"]
        y_true.append(true)
        f_name = row["filename"]
        image_loc = os.path.join(config.DATA_DIR, f_name)
        image = cv.imread(image_loc)
        pred = n_squares(image)
        y_pred.append(pred)
        if pred != true:
            print(pred, image_loc)
    precision = precision_score(y_true, y_pred, average="micro")
    print(precision)
    return precision

if __name__ == "__main__":
    main()