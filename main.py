import cvutils.camera as cam
import cvutils.capture as cap
import cvutils.recon as rc
from configs import Config
import matplotlib.pyplot as plt

def main(args=[]):
    L = cam.Camera(Config.cam, "L")
    R = cam.Camera(Config.cam, "R")
    capL = cap.Image("./samples/cap_L.png", L)
    capR = cap.Image("./samples/cap_R.png", R)
    recon = rc.reconstruct(capR, capL, 7)
    X = []
    Y = []
    Z = []
    ax = plt.axes(projection ="3d")
    for u in recon:
        u_x, u_y, u_z = u.A1
        X.append(u_x)
        Y.append(-u_y)
        Z.append(u_z)
    ax.scatter(X, Z, Y, color="red")
    ax.set_xlabel('x', fontweight ='bold')
    ax.set_ylabel('z', fontweight ='bold')
    ax.set_zlabel('y', fontweight ='bold')
    plt.show()
    return 0
