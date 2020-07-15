"""The full eval script."""
import os
from tqdm import tqdm
import pandas as pd
import cv2 as cv 
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score

from .data_loader import CaptchaDataset
from ..features.n_squares import n_squares
from .predict import predict
from .. import config


def eval_cv():
    "Evaluates the square counting method on the full data"
    # read in the full data
    df_all = pd.read_csv("data/all.csv")
    # make arrays to store the data
    y_pred, y_true = [0]*len(df_all), [0]*len(df_all)
    print("Incorrect predictions: ")
    for index, row in df_all.iterrows():
        y_true[index] = row["target"]
        f_name = row["filename"]
        # where the image is located
        image_loc = os.path.join(config.RAW_DIR, f_name)
        # load the image
        image = cv.imread(image_loc)
        # save the openCV prediction
        y_pred[index] = n_squares(image)
        # print
        if pred != true:
            print("\t", pred, image_loc)
    # return the result
    return precision_score(y_true, y_pred, average="micro")

def dl():
    """Evaluate the dl model on the test set"""
    # the test dataset
    eval_ds = CaptchaDataset(
        csv_file=config.TEST_DATA, 
        root_dir=config.PROC_DIR, 
        transform=transforms.ToTensor()
    )
    # the test data loader
    eval_loader = DataLoader(eval_ds)
    # storing results
    y_true, y_pred = [0]*len(eval_ds), [0]*len(eval_ds)
    # evaluate the model on the test data
    for i, (data, target) in tqdm(enumerate(eval_loader), total=len(eval_ds)):
        res = predict(data)
        y_true[i] = target
        y_pred[i] = res
    return precision_score(y_true, y_pred, average="micro")


if __name__ == "__main__":
    score = dl()
    print("Precision", score)