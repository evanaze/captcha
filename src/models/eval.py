"""The full eval script."""
import os
from tqdm import tqdm
import pandas as pd
import argparse
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
    print("Evaluating OpenCV method")
    for index, row in tqdm(df_all.iterrows(), total=len(df_all)):
        # save the true value
        true = row["target"]
        y_true[index] = true
        # where the image is located
        f_name = row["filename"]
        image_loc = os.path.join(config.RAW_DIR, f_name)
        # load the image
        image = cv.imread(image_loc)
        # save the openCV prediction
        pred = n_squares(image)
        y_pred[index] = pred
    # return the result
    return precision_score(y_true, y_pred, average="micro")


def eval_dl(model="models/captcha_dcnn.pt"):
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
    print("Evaluating DCNN")
    for i, (data, target) in tqdm(enumerate(eval_loader), total=len(eval_ds)):
        # save the prediction and the true value
        y_true[i], y_pred[i] = target, predict(data, model)
    # return the precision score
    return precision_score(y_true, y_pred, average="micro")


def main():
    """Uses argparse to intelligently run eval"""
    parser = argparse.ArgumentParser(description='Captcha evaluation')
    parser.add_argument('--square', '-s', action='store_true',
                    help='whether or not to evaluate the OpenCV method')
    parser.add_argument('--dcnn', '-d', action='store_true',
                    help='evaluate the DCNN model')
    args = parser.parse_args()
    # if we should evaluate the opencv model
    if args.square:
        score = eval_cv()
        print("OpenCV precision:", score)
    # if we should evaluate the dcnn model
    if args.dcnn:
        score = eval_dl()
        print("DCNN precision:", score)


if __name__ == "__main__":
    main()