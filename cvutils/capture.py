import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# TODO: write static image class
class Image:
    def __init__(self, loc, cam):
        self.im = plt.imread(loc)
        self.cam = cam

    def corners(self, max=500, q=0.01, dist=5):
        gr = cv.cvtColor(self.im, cv.COLOR_RGB2GRAY)
        gr = np.float32(gr)
        corners = cv.goodFeaturesToTrack(gr, max, q, dist)
        corners = np.int0(corners)
        return corners
