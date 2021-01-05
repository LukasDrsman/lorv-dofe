import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


class Image:
    def __init__(self, loc, cam):
        self.im = cv.imread(loc)
        self.cam = cam


def surfaceMasks(im, k):
    h, w, cs = im.shape
    Z = im.reshape((-1, cs))
    Z = np.float32(Z)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center=cv.kmeans(Z, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    masks = res.reshape((im.shape))
    return masks

def separateMasks(masks):
    h, w, cs = masks.shape
    uniques = np.unique(masks.reshape(w*h, cs), axis=0)
    smasks = []
    for u in uniques:
        mask = np.zeros((h, w), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                if(all(masks[y, x] == u)):
                    mask[y, x] = 255
        smasks.append(mask)
    return smasks

def applyMasks(im, smasks):
    surfaces = []
    h, w, cs = im.shape
    for mask in smasks:
        surface = np.zeros((h, w, 4), dtype=np.uint8)
        ay, ax = np.where(mask == 255)
        for n in range(len(ay)):
            x = ax[n]
            y = ay[n]
            col = im[y, x]
            b = col[0]; g = col[1]; r = col[2]
            surface[y, x] = [b, g, r, 255]
        surfaces.append(surface)
    return surfaces

def edgesMask(mask):
    edges = cv.Canny(mask, 50, 50)
    return edges

def cleanMask(mask, thresh):
    h, w = mask.shape
    nmask = np.zeros((h, w), dtype=np.uint8)
    ay, ax = np.where(mask == 255)
    for n in range(len(ay)):
        y = ay[n]
        x = ax[n]
        an = 0
        for i in range(-1, 2):
            for j in range(-1, 2):
                if(h > (i + y) and w > (j + x)):
                    an += (mask[i + y, j + x] == 255) * 1
                    if(an > thresh):
                        nmask[y, x] = 255
                        break
    return nmask
