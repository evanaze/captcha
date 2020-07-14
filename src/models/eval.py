"""The full eval script."""
import os
import pandas as pd
from sklearn.metrics import precision_score
from torchvision import transforms
import cv2 as cv 
from torch.utils.data import DataLoader

from .data_loader import CaptchaDataset
from ..features.n_squares import n_squares
from .predict import predict
from .. import config


def square_counting():
    "Evaluates the square counting method on the full data"
    # read in the full data
    df_all = pd.read_csv("data/all.csv")
    y_pred, y_true = [], []
    for index, row in df_all.iterrows():
        true = row["target"]
        y_true.append(true)
        f_name = row["filename"]
        image_loc = os.path.join(config.RAW_DIR, f_name)
        image = cv.imread(image_loc)
        pred = n_squares(image)
        y_pred.append(pred)
        if pred != true:
            print(pred, image_loc)
    precision = precision_score(y_true, y_pred, average="micro")
    print(precision)
    return precision

def dl():
    """Evaluate the dl model on the test set"""
    transform = transforms.toTensor()
    eval_ds = CaptchaDataset(
        csv_file=config.TEST_DATA, 
        root_dir=config.PROC_DIR, 
        transform=transform
    )
    eval_loader = DataLoader(
        eval_ds
    )
    y_true, y_pred = [0]*len(eval_ds), [0]*len(eval_ds)
    for i, (data, target) in enumerate(data_loader)
        res = predict(data)
        y_true[i] = target
        y_pred[i] = res
    return precision_score(y_true, y_pred, average="micro")


if __name__ == "__main__":
    dl()