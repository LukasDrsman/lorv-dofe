import cv2 as cv
import numpy as np
import camera as cam

def main(args=[]):
    lcam = cam.Camera(open("/home/lukas/Projects/lorv-dofe/cam.json"), "L-cam")
    rcam = cam.Camera(open("/home/lukas/Projects/lorv-dofe/cam.json"), "R-cam")
    concat = np.concatenate((lcam.cap, rcam.cap), axis=1)
    cv.imshow("captures", concat)
    while True:
        if cv.waitKey(1) == ord('q'):
            break
    cv.destroyAllWindows()
    return 0
