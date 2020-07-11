""" Preprocess the data by Laplace and thresholding
"""
import numpy as np 
import argparse
import cv2 as cv
from .. import config


def preprocess(image):
    """Takes in an image, and returns the de-noised image"""
    # compute laplacian
    laplacian = cv.Laplacian(image, cv.CV_64F)
    # choose first channel of the laplacian
    squares_channel = cv.split(laplacian)[0].astype(np.uint8)
    # threshold out extra noise
    thresh = cv.threshold(squares_channel, 60, 255, cv.THRESH_BINARY)[1]
    # standardize the image size
    src = cv.resize(thresh, (300, 300), interpolation=cv.INTER_AREA)
    return src

def main():
    """A way to observe the preprocessing"""
    # parse the image from the command line
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to the input image")
    args = ap.parse_args()
    # read in the raw image
    image_loc = config.RAW_DIR + "/" + args.image
    image = cv.imread(image_loc)
    # show the processed image
    cv.imshow("processed image", preprocess(image))
    cv.waitKey()


if __name__ == "__main__":
    main()