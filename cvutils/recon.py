import numpy as np

def triangulate(v1, v2, P1, P2):
    v1 = v1.A1
    v2 = v2.A1
    v1_x, v1_y = v1
    v2_x, v2_y = v2

    if(v1_x != v2_x):
        A = np.array(
            [[P1.f_x, -( v1_x - P1.p_x )],
             [P2.f_x, -( v2_x - P2.p_x )]]
        )
        B = np.array(
            [-P1.C[2] * ( v1_x - P1.p_x ) + P1.C[0] * P1.f_x,
             -P2.C[2] * ( v2_x - P2.p_x ) + P2.C[0] * P2.f_x]
        )

        u_x, u_z = (np.linalg.inv(A).dot(B))
        u_y = ( u_z * ( v1_y - P1.p_y ) - P1.C[2] * ( v1_y - P1.p_y ) ) / P1.f_y + P1.C[1]

        u = np.matrix(
            [[u_x],
             [u_y],
             [u_z]]
        )

    elif(v1_y != v2_y):
        A = np.array(
            [[P1.f_y, -( v1_y - P1.p_y )],
             [P2.f_y, -( v2_y - P2.p_y )]]
        )
        B = np.array(
            [-P1.C[2] * ( v1_y - P1.p_y ) + P1.C[1],
             -P2.C[2] * ( v2_y - P2.p_y ) + P2.C[1]]
        )

        u_y, u_z = (np.linalg.inv(A).dot(B))
        u_x = ( u_z * ( v1_x - P1.p_x ) - P1.C[2] * ( v1_x - P1.p_x ) ) / P1.f_x + P1.C[0]

        u = np.matrix(
            [[u_x],
             [u_y],
             [u_z]]
        )

    else:
        raise Exception("v1 == v2")

    return u

# TODO: write a reconstruction function
def reconstruct(cap1, cap2, max=500, q=0.01, dist=5):
    recon = []
    im1 = cap1.im
    im2 = cap2.im
    h1, w1, trash = im1.shape
    h2, w2, trash = im2.shape
    P1 = cap1.cam
    P2 = cap2.cam
    C1 = cap1.corners(max, q, dist)
    C2 = cap2.corners(max * 2, q, dist)
    for corner1 in C1:
        c1_x, c1_y = corner1[0]
        c1 = np.matrix([[c1_x], [c1_y]])
        c1_r, c1_g, c1_b, c1_a = im1[c1_y, c1_x]
        if(c1_x != P1.p_x or c1_y != P1.p_y):
            n1 = P1.f_x + 2
            n2 = P1.f_x + 6
            t1 = P1.projLine(c1, n1)
            t2 = P1.projLine(c1, n2)
            E = []
            if(t1[0,0] != t2[0,0]):
                for corner2 in C2:
                    c2_x, c2_y = corner2[0]
                    c2 = np.matrix([[c2_x], [c2_y]])
                    c2_r, c2_g, c2_b, c2_a = im2[c2_y, c2_x]
                    e_x, e_y = (P2.epiLine(t1, t2, c2_x)).A1
                    grade = (abs(e_y - c2_y)) + (abs(c2_r - c1_r)) + (abs(c2_g - c1_g)) + (abs(c2_b - c1_b))
                    E.append((grade, c2))

            elif(t1[1,0] != t2[1,0]):
                for corner2 in C2:
                    c2_x, c2_y = corner2[0]
                    c2 = np.matrix([[c2_x], [c2_y]])
                    c2_r, c2_g, c2_b, c2_a = im2[c2_y, c2_x]
                    e_x, e_y = (P2.epiLine(t1, t2, c2_y)).A1
                    grade = (abs(e_x - c2_x)) + (abs(c2_r - c1_r)) + (abs(c2_g - c1_g)) + (abs(c2_b - c1_b))
                    E.append((grade, c2))

            else:
                continue

            E.sort(key=lambda grade: grade[0])
            recon.append(
                triangulate(c1, E[0][1], P1, P2)
            )

        else:
            continue

    return recon
