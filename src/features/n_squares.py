import argparse
import numpy as np
import matplotlib.pyplot as plt
import imutils
import cv2 as cv 
from .shape_detector import ShapeDetector
from .preprocess import preprocess

""" Uses OpenCV to count the number of squares in a Captcha

"""


def n_squares(image, display=False):
    image = preprocess(image)
    if display:
        plt.imshow(image)
        plt.show()
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
        if a > 1150 and a < 1900:
            n_squares += 2
        elif a > 1900 and a < 3000:
            n_squares += 3
        elif a > 3000:
            n_squares += 4
        elif shape == "square":
            n_squares += 1
    return int(n_squares)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to the input image")
    args = vars(ap.parse_args())
    image = cv.imread(args["image"])
    print(n_squares(image, True))

if __name__ == "__main__":
    main()