import cv2 as cv
import numpy as np
import json

class Camera:
    def __init__(self, conf, name):
        self.cam = json.loads(conf.read())["Camera"]            # load camera config --> cam
        self.name = name
        self.inst = self.cam["Instances"][self.name]            # get the instance of our camera
        self.model = self.cam["Models"][self.inst["model"]]     # get the model of our camera
        self.K = np.matrix(self.model["K"])                     # get the intrinsc parameters matrix
        self.Rt = np.matrix(self.inst["Rt"])                    # get the extrinsic parameters matrix
        self.P = np.matmul(self.K, self.Rt)                     # calculate the camera matrix
        self.cap = cv.imread(self.inst["cap"], 1)
