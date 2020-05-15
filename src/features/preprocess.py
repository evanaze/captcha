import numpy as np 
import argparse
import matplotlib.pyplot as plt
import cv2 as cv

""" Preprocess the data by Laplace and thresholding

"""

def preprocess(image):
    """Takes in an image, and returns the de-noised image"""
    laplacian = cv.Laplacian(image, cv.CV_64F)
    squares_channel = cv.split(laplacian)[0].astype(np.uint8)
    thresh = cv.threshold(squares_channel, 60, 255, cv.THRESH_BINARY)[1]
    src = cv.resize(thresh, (300, 300), interpolation=cv.INTER_AREA)
    return src

def main():
    """A way to observe the preprocessing"""
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to the input image")
    args = vars(ap.parse_args())
    image = cv.imread(args["image"])
    plt.imshow(preprocess(image))
    plt.show()


if __name__ == "__main__":
    main()