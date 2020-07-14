""" The overall predict script for the full model
"""
import os
import argparse

from skimage import io
import torch

from .model import Net
from ..features.preprocess import preprocess
from .. import config


def predict(file_name=None):
    "Predict on an image from a dataloader"
    # load the model
    model = Net()
    model.load_state_dict(torch.load("models/captcha_cnn_f0.pt"))
    model.eval()
    # get the output of the model
    output = model() 
    # choosing the model's result
    res = output.argmax(dim=1, keepdim=True).numpy()[0][0] + 1 
    return true, res

def main():
    parser = argparse.ArgumentParser(description='Captcha evaluation')
    parser.add_argument('--file-name', type=str, default="8_1587401157.16.png", metavar='N',
                        help='the location of the file to evaluate')
    args = parser.parse_args()
    true, res = predict(args.file_name)
    print(f"True: {true}, Predicted: {res}")

if __name__ == "__main__":
    main()
    