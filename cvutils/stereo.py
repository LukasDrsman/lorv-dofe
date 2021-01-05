import numpy as np
import cv2 as cv

class Stereo:
    def __init__(self, cap1, cap2):
        self.cap1 = cap1
        self.cap2 = cap2

    def findMatch(self, chunk, u):
        cam1 = self.cap1.cam
        cam2 = self.cap2.cam
        im1 = self.cap1.im
        im2 = self.cap2.im
        p1, p2 = (
            cam1.projLine(u, (cam1.C[2] + cam1.f)),
            cam1.projLine(u, (cam1.C[2] + cam1.f)*2)
        )

        offset = int((chunk - 1) / 2)

        h, w, cs = im2.shape

        ch1 = np.zeros((chunk, chunk, cs), np.uint8)
        n, m = 0, 0
        px, py = int(u.A1[0]), int(u.A1[1])
        for i in range(-offset, offset + 1):
            for j in range(-offset, offset + 1):
                ch1[n,m] = im1[i + py, j + px]
                n += 1
            m += 1
            n = 0

        matches = []
        for x in range(offset, w - offset):
            c, d = cam2.epiLine(p1, p2, x).A1
            c = int(c)
            d = int(d)
            ch2 = np.zeros((chunk, chunk, cs), np.uint8)
            n = 0
            m = 0
            for i in range(-offset, offset + 1):
                for j in range(-offset, offset + 1):
                    ch2[n,m] = im2[i + d, j + c]
                    n += 1
                m += 1
                n = 0
            diff = np.sum(abs(ch1 - ch2))
            matches.append( (diff, (c, d)) )
        matches.sort(key=lambda x: x[0])
        return matches[0][1]

    def triangulate(self, v1, v2):
        cam1 = self.cap1.cam
        cam2 = self.cap2.cam
        v1 = v1.A1
        v2 = v2.A1
        v1_x, v1_y = v1
        v2_x, v2_y = v2

        if(v1_x != v2_x):
            A = np.array(
                [[cam1.f_x, -( v1_x - cam1.p_x )],
                 [cam2.f_x, -( v2_x - cam2.p_x )]]
            )
            B = np.array(
                [-cam1.C[2] * ( v1_x - cam1.p_x ) + cam1.C[0] * cam1.f_x,
                 -cam2.C[2] * ( v2_x - cam2.p_x ) + cam2.C[0] * cam2.f_x]
            )

            u_x, u_z = (np.linalg.inv(A).dot(B))
            u_y = ( u_z * ( v1_y - cam1.p_y ) - cam1.C[2] * ( v1_y - cam1.p_y ) ) / cam1.f_y + cam1.C[1]

            u = np.matrix(
                [[u_x],
                 [u_y],
                 [u_z]]
            )

        elif(v1_y != v2_y):
            A = np.array(
                [[cam1.f_y, -( v1_y - cam1.p_y )],
                 [cam2.f_y, -( v2_y - cam2.p_y )]]
            )
            B = np.array(
                [-cam1.C[2] * ( v1_y - cam1.p_y ) + cam1.C[1],
                 -cam2.C[2] * ( v2_y - cam2.p_y ) + cam2.C[1]]
            )

            u_y, u_z = (np.linalg.inv(A).dot(B))
            u_x = ( u_z * ( v1_x - cam1.p_x ) - cam1.C[2] * ( v1_x - cam1.p_x ) ) / cam1.f_x + cam1.C[0]

            u = np.matrix(
                [[u_x],
                 [u_y],
                 [u_z]]
            )

        else:
            raise Exception("v1 == v2")

        return u
