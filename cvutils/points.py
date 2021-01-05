# [WIP] methods in this module are extremely inefficient and inaccurate compared to the
# methods cv2
import numpy as np
import cv2 as cv

def notSoFAST(im, N, t):
    h, w = im.shape
    corners = []
    for y in range(3, h - 3):
        for x in range(3, w - 3):
            p = im[y, x]
            bc = [
                im[y - 3, x], im[y - 3, x + 1],
                im[y - 2, x + 2], im[y - 1, x + 3],
                im[y, x + 3], im[y + 1, x + 3],
                im[y + 2, x + 2], im[y + 3, x + 1],
                im[y + 3, x], im[y + 3, x - 1],
                im[y + 2, x - 2], im[y + 1, x - 3],
                im[y, x - 3], im[y - 1, x - 3],
                im[y - 2, x - 2], im[y - 3, x - 1]
            ]

            bc = np.array(bc + bc[0:13])
            high = bc > (p + t)
            if((np.where(high == True)[0]).shape[0] >= N):
                hn = any(np.array([e.tolist().count(True) for e in (np.split(high, np.where(high[:] == False)[0]))]) >= N)
                if(hn):
                    corners.append((x,y))
                    continue
            low = bc < (p - t)
            if((np.where(low == True)[0]).shape[0] >= N):
                ln = any(np.array([e.tolist().count(True) for e in (np.split(low, np.where(low[:] == False)[0]))]) >= N)
                if(ln):
                    corners.append((x,y))
                    continue

    return corners

def FASTalong(im, N, t, along):
    corners = []
    ay, ax = np.where(along == 255)
    for y, x in zip(ay, ax):
        p = im[y, x]
        bc = [
            im[y - 3, x], im[y - 3, x + 1],
            im[y - 2, x + 2], im[y - 1, x + 3],
            im[y, x + 3], im[y + 1, x + 3],
            im[y + 2, x + 2], im[y + 3, x + 1],
            im[y + 3, x], im[y + 3, x - 1],
            im[y + 2, x - 2], im[y + 1, x - 3],
            im[y, x - 3], im[y - 1, x - 3],
            im[y - 2, x - 2], im[y - 3, x - 1]
        ]

        bc = np.array(bc + bc)
        low = bc < (p - t)
        if((np.where(low == True)[0]).shape[0] >= N):
            ln = any(np.array([e.tolist().count(True) for e in (np.split(low, np.where(low[:] == False)[0]))]) >= N)
            if(ln):
                corners.append((x,y))
                continue
        high = bc > (p + t)
        if((np.where(high == True)[0]).shape[0] >= N):
            hn = any(np.array([e.tolist().count(True) for e in (np.split(high, np.where(high[:] == False)[0]))]) >= N)
            if(hn):
                corners.append((x,y))
                continue


    return corners
