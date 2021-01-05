# math and cv tools
import numpy as np
import cv2 as cv

# cvutils imports
import cvutils.camera as cam
import cvutils.capture as cap
import cvutils.stereo as st

# json configs
from configs import Config

# visualization modules
import vis3dpy.plot as vis
import matplotlib.pyplot as plt


def main(args=[]):
    # Usefull stuff
    capL = cap.Image("./samples/cap_L.png", cam.Camera(Config.cam, "L"))
    capR = cap.Image("./samples/cap_R.png", cam.Camera(Config.cam, "R"))
    stereo = st.Stereo(capL, capR)

    # Interest points
    grayL = cv.cvtColor(capL.im, cv.COLOR_BGR2GRAY)
    fast = cv.FastFeatureDetector_create()
    cornersL = fast.detect(grayL, None)
    pointsL = []
    for point in cv.KeyPoint_convert(cornersL):
        pointsL.append(point)

    # Correspondence
    corr = []
    for x, y in pointsL:
        u = np.matrix([[x], [y]])
        corr.append([(x, y), stereo.findMatch(3, u)])

    # Triangulation and visualization
    fig = vis.Figure(800, 800, (50, 50, 50))
    for p1, p2 in corr:
        x1, y1 = p1
        x2, y2 = p2
        v1 = np.matrix([[x1], [y1]])
        v2 = np.matrix([[x2], [y2]])
        w = stereo.triangulate(v1, v2)
        print(w)
        wx, wy, wz = w.A1
        col = capL.im[int(y1), int(x1)]
        fig.scatter([[wx/1000, -wy/1000, -wz/1000]], col)

    fig.show()
    
    return 0
