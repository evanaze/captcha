import os
import pandas as pd
from sklearn.metrics import f1_score
from .predict import predict
from . import config

def main():
    """Evaluate on the test set"""
    y_true, y_pred = [], []
    test = pd.read_csv("input/test.csv")
    for f in test.filename:
        true, res = predict(f)
        y_true.append(true)
        y_pred.append(res)
    return f1_score(y_true, y_pred, average="micro")

if __name__ == "__main__":
    print(main())