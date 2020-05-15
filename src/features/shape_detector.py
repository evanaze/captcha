# import the necessary packages
import cv2

class ShapeDetector:
    def __init__(self):
        pass
    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        bb_area = w * h
        if ar > 10: 
            shape = "line"
        elif len(approx) == 4 or ar >= 0.90 and ar <= 1.10:
            shape = "square"
        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"
        # return the name of the shape
        return shape, bb_area