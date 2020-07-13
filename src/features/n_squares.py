""" Uses OpenCV to count the number of squares in a Captcha
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import imutils
import cv2 as cv 

from .shape_detector import ShapeDetector
from .preprocess import preprocess
from .. import config


def n_squares(image, display=False):
    image = preprocess(image)
    if display:
        cv.imshow("input image", image)
        cv.waitKey()
    cnts = cv.findContours(image.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    sd = ShapeDetector()
    n_squares = 0
    for c in cnts:
        # compute the center of the contour 
        M = cv.moments(c)
        a = M["m00"]
        if a < 50:
            continue
        shape, bb_area = sd.detect(c)
        diff = bb_area - a
        if display:
            print(a, diff, shape)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # draw the contour and center of the shape on the image
            cv.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv.circle(image, (cX, cY), 7, (255, 255, 255), -1)
            cv.putText(image, "center", (cX - 20, cY - 20),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # show the image
            cv.imshow("Image", image)
            cv.waitKey(0)
        else:
            pass
        # the size thresholds for our heuristic
        c1, c2, c3 = 1800, 3500, 5000
        if a > c1 and a < c2:
            n_squares += 2
        elif a > c2 and a < c3:
            n_squares += 3
        elif a > c3:
            n_squares += 4
        elif shape == "square":
            n_squares += 1
    return int(n_squares)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to the input image")
    args = vars(ap.parse_args())
    # parse the image from the command line
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to the input image")
    args = ap.parse_args()
    # read in the raw image
    image_loc = config.RAW_DIR + "/" + args.image
    image = cv.imread(image_loc)
    # show the processed image
    print(n_squares(image, True))

if __name__ == "__main__":
    main()