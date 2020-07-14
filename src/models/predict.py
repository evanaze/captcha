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
    "This is for you to write"
    # Load the image
    true = int(file_name.split('_')[0])
    file_loc = os.path.join(config.RAW_DIR, file_name)
    image = io.imread(file_loc)
    # predict
    model = Net()
    model.load_state_dict(torch.load("models/captcha_cnn_f0.pt"))
    model.eval()
    processed = preprocess(image)
    output = model(transforms.ToTensor(processed)) # unsqeezing to get in the proper format (1, 3, 200, 200)
    res = output.argmax(dim=1, keepdim=True).numpy()[0][0] + 1 # choosing the model's result
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
    