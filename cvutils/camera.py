import numpy as np
import json
from math import cos, sin, radians

class Camera:
    def __init__(self, conf, id):
        # load configs
        conf = json.loads(conf)["Camera"]
        instance = conf["Instances"][id]
        model = conf["Models"][instance["model"]]
        # camera parameters
        self.C = instance["C"]
        self.I_W, self.I_H = model["w_h"]
        self.S_W, self.S_H = model["W_H"]
        self.W, self.H = ((self.S_W/self.I_W), (self.S_H/self.I_H))
        self.f = model["f"]
        self.f_x = self.f / self.W
        self.f_y = self.f / self.H
        self.p_x = self.I_W / 2
        self.p_y = self.I_H / 2
        self.rot = instance["R-XYZ"]
        al, be, ga = self.rot
        al = radians(al)
        be = radians(be)
        ga = radians(ga)
        # 3x3 orthogonal rotation matrix - Tait-Bryan X_al Y_be Z_ga angles
        R = np.matrix(
            [[cos(be) *cos(ga), -cos(be) *sin(ga), sin(be)],
            [cos(al) *sin(ga) + cos(ga) *sin(al) *sin(be), cos(al) *cos(ga) - sin(al) *sin(be) *sin(ga), -cos(be) *sin(al)],
            [sin(al) *sin(ga) - cos(al) *cos(ga) *sin(be), cos(ga) *sin(al) + cos(al) *sin(be) *sin(ga), cos(al) *cos(be)]]
        )
        # 3x3 intrinsic camera parameters matrix
        K = np.matrix(
            [[ self.f_x,  0,         self.p_x ],
             [ 0,         self.f_y,  self.p_y ],
             [ 0,         0,         1        ]]
        )
        # 3x3 inverse orthogonal rotation matrix
        iR = np.linalg.inv(R)
        # rotated translation vector
        t = np.multiply(iR, np.matrix(
            [[self.C[0]],
             [self.C[1]],
             [self.C[2]]]
        )).A1
        # 3x4 projection matrix - P = K (R^T    -R^T*c)
        self.P = np.matmul(K, np.matrix(
                [[iR[0,0], iR[0,1], iR[0,2], t[0]],
                 [iR[1,0], iR[1,1], iR[1,2], t[1]],
                 [iR[2,0], iR[2,1], iR[2,2], t[2]]]
            )
        )

    # poject u onto image plane
    def project(self, w):
        u = np.matmul(self.P, w)
        return np.delete(np.multiply(( 1 / u.A1[2] ), u), 2, 0)

    def projLine(self, u, z, h=1):
        u_x, u_y = u.A1
        v_z = self.C[2] + self.f
        P = self.P
        # compute y component of vector v
        v_y = ( ( ( u_x*P[2,0] - P[0,0] )*( v_z*( P[1,2] -u_y*P[2,2] ) + P[1,3] -u_y*P[2,3] ) - ( u_y*P[2,0] - P[1,0] )*( v_z*( P[0,2] -u_x*P[2,2] ) + P[0,3] -u_x*P[2,3] ) )
              / ( ( P[0,1] -u_x*P[2,1] )*( u_y*P[2,0] - P[1,0] ) - ( P[1,1] -u_y*P[2,1] )*( u_x*P[2,0] - P[0,0] ) ) )
        # compute x component of vector v
        v_x = ( ( v_y*( P[0,1] -u_x*P[2,1] ) + v_z*( P[0,2] -u_x*P[2,2] ) + P[0,3] -u_x*P[2,3] )
              / ( u_x*P[2,0] - P[0,0] ) )
        # find x and z components of a vector on line with given z component
        lam = ( z - self.C[2] ) / self.f
        x = self.C[0] + ( lam * ( v_x - self.C[0] ) )
        y = self.C[1] + ( lam * ( v_y - self.C[1] ) )
        if(h == 1):
            return np.matrix(
                [[x],
                 [y],
                 [z],
                 [1]]
            )
        else:
            return np.matrix(
                [[x],
                 [y],
                 [z]]
            )

    def epiLine(self, t1, t2, n):
        g1_x, g1_y = self.project(t1).A1
        g2_x, g2_y = self.project(t2).A1
        # calculate location of a point on an epipolar line when epipolar point 1 x != point 2 x
        if(g1_x != g2_x):
            lam = ( n - g1_x ) / ( g1_x - g2_x )
            y = lam * ( g1_y - g2_y ) + g1_y
            e = np.matrix(
                [[n],
                 [y]]
            )
            return e
        # calculate location of a point on an epipolar line when epipolar point 1 y != point 2 y
        elif(g1_y != g2_y):
            lam = ( n - g1_y ) / ( g1_y - g2_y )
            x = lam * ( g1_x - g2_x ) + g1_x
            e = np.matrix(
                [[x],
                 [n]]
            )
            return e

        else:
            raise Exception("t1 == t2")
