import numpy as np

def is_adj(domain1, domain2):
    d1 = np.roll(domain1, -1, axis=0)
    d2 = np.roll(domain1, 1, axis=0)
    d3 = np.roll(domain1, -1, axis=1)
    d4 = np.roll(domain1, 1, axis=1)
    d1[:,0] = 0
    d2[:,-1] = 0
    d3[0] = 0
    d4[-1] = 0
    d5 = np.roll(d1, -1, axis=1)
    d6 = np.roll(d1, 1, axis=1)
    d7 = np.roll(d2, -1, axis=1)
    d8 = np.roll(d2, 1, axis=1)
    d5[0] = 0
    d6[-1] = 0
    d7[:,0] = 0
    d8[:,-1] = 0
    d1 += domain2
    d2 += domain2
    d3 += domain2
    d4 += domain2
    d5 += domain2
    d6 += domain2
    d7 += domain2
    d8 += domain2
    return len(d1[d1 >= 2]) != 0 or len(d2[d2 >= 2]) != 0 or len(d3[d3 >= 2]) != 0 or len(d4[d4 >= 2]) != 0 \
        or len(d5[d5 >= 2]) != 0 or len(d6[d6 >= 2]) != 0 or len(d7[d7 >= 2]) != 0 or len(d8[d8 >= 2]) != 0