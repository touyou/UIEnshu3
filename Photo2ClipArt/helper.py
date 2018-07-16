import numpy as np
import cv2

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

def images(node, idx, file):
    gen_img = np.zeros(node.x[0].domain.shape, np.uint8)
    for l in node.x:
        # l_img = np.tile(l.c0, (gen_img.shape[0], gen_img.shape[1], 1))
        # gen_img += l_img * l.domain + gen_img * (1 - l.domain)
        gen_img += (l.get_color_mat().astype(np.float64) * l.get_alpha_mat() + (1 - l.get_alpha_mat()) * gen_img.astype(np.float64)).astype(np.uint8)
    bgr = cv2.split(gen_img)
    bgra = cv2.merge(bgr + [node.visited_domain()[:,:,0] * 255])
    cv2.imwrite("old/res_test/{}_result_{}_{}.png".format(file, idx, node.reward), bgra)
    for i in range(len(node.children)):
        images(node.children[i], idx + str(i), file)
        
def norm_color(c):
    if c < 0:
        return 0
    if c > 255:
        return 255
    return c

def norm_alpha(a):
    if a < 0:
        return 0.0
    if a > 1.0:
        return 1.0
    return a