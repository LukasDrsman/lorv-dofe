import numpy as np
import json

class Camera:
    def __init__(self, conf, id):

        # load configs
        self.conf = json.loads(conf)["Camera"]
        self.instance = self.conf["Instances"][id]
        self.model = self.conf["Models"][self.instance["model"]]

        # camera parameters
        self.C = self.instance["C"]
        self.w = self.model["w_h"][0]
        self.h = self.model["w_h"][1]
        self.W = self.model["W_H"][0]
        self.H = self.model["W_H"][1]
        self.f = self.model["f"]
        self.f_x = ( self.w * self.f ) / self.W
        self.f_y = ( self.h * self.f ) / self.H
        self.p_x = self.w / 2
        self.p_y = self.h / 2
        self.t_x = -( self.f_x * self.C[0] ) - self.p_x * self.C[2]
        self.t_y = -( self.f_y * self.C[1] ) - self.p_y * self.C[2]
        self.t_z = -( self.C[2] )

        # camera projection parameters matrix
        self.P = np.matrix(
            [[self.f_x, 0, self.p_x, self.t_x],
             [0, self.f_y, self.p_y, self.t_y],
             [0, 0, 1, self.t_z]]
        )

    # poject u onto image plane
    def project(self, u):
        v_s = np.matmul(self.P, u)
        v = np.multiply(( 1 / v_s.A1[2] ), v_s)
        return np.delete(v, 2, 0)

    def projLine(self, v, n):
        v_x, v_y = v.A1
        if(self.p_x != v_x):
            z = ( -self.C[2] * ( v_x - self.p_x ) + self.f_x * ( self.C[0] - n ) ) / ( self.p_x - v_x )
            y = - ( ( ( self.p_y - v_y ) * ( self.f_x * ( self.C[0] - n ) - self.C[2] * ( v_x - self.p_x ) ) ) / ( self.f_y * ( self.p_x - v_x ) ) )
            - ( ( self.C[2] * ( v_y - self.p_y ) ) / self.f_y )
            + self.C[1]
            t = np.matrix(
                [[n],
                [y],
                [z],
                [1]]
            )
            return t

        elif(self.p_y != v_y):
            z = ( -self.C[2] * ( v_y - self.p_y ) + self.f_y * ( self.C[1] - n ) ) / ( self.p_y - v_y )
            x = - ( ( ( self.p_x - v_x ) * ( self.f_y * ( self.C[1] - n ) - self.C[2] * ( v_y - self.p_y ) ) ) / ( self.f_x * ( self.p_y - v_y ) ) )
            - ( ( self.C[2] * ( v_x - self.p_x ) ) / self.f_x )
            + self.C[0]
            t = np.matrix(
                [[x],
                [n],
                [z],
                [1]]
            )
            return t

        else:
            raise Exception("singularity point")

    def epiLine(self, t1, t2, n):
        g1_x, g1_y = self.project(t1).A1
        g2_x, g2_y = self.project(t2).A1

        if(g1_x != g2_x):
            lam = ( n - g1_x ) / ( g1_x - g2_x )
            y = lam * ( g1_y - g2_y ) + g1_y
            e = np.matrix(
                [[n],
                 [y]]
            )
            return e

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
