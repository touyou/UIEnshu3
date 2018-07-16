import numpy as np
import copy
import cv2

'''
基本式の係数だけ持っていて、計算することであるピクセルの値が返るようにする（？）
'''
class Layer:
    
    def __init__(self, c0, c1, a0, a1, oc, oa, domain):
        self.c0 = c0
        self.c1 = c1
        self.a0 = a0
        self.a1 = a1
        self.oc = oc
        self.oa = oa
        self.domain = domain
        self.alpha_mat = np.zeros(self.domain.shape)
        for iy in range(len(self.domain)):
            for ix in range(len(self.domain[0])):
                if self.domain[iy, ix, 0] == 0:
                    continue
                a = self.alpha(np.array([iy, ix]).T)
                self.alpha_mat[iy, ix] = np.array([a, a, a])
        # print(len(self.alpha_mat[self.alpha_mat != 0]))
        self.color_mat = np.zeros(self.domain.shape, dtype=np.uint8)
        for iy in range(len(self.domain)):
            for ix in range(len(self.domain[0])):
                if self.domain[iy, ix, 0] == 0:
                    continue
                self.color_mat[iy, ix] = self.color(np.array([iy, ix]).T)
        # bgr= cv2.split(self.color_mat)
        # abgr = cv2.split(self.alpha_mat)
        # print(bgr)
        # print(abgr)
        # alpha = abgr[0] * 255
        # bgra = cv2.merge([bgr[0], bgr[1], bgr[2], alpha.astype(np.uint8)])
        # print(bgra)
        # exit()
        # test = self.color_mat.astype(np.float64) * self.alpha_mat - self.color_mat.astype(np.float64)
        # self.color_mat = None
        # self.alpha_mat = None

    def alpha(self, p):
        return self.a0 + self.a1 * self.oa.dot(p)
    
    def color(self, p):
        return self.c0 + self.c1 * self.oc.dot(p)
    
    def get_alpha_mat(self):
        if self.alpha_mat is None:
            self.alpha_mat = np.zeros(self.domain.shape)
            for iy in range(len(self.domain)):
                for ix in range(len(self.domain[0])):
                    if self.domain[iy, ix, 0] == 0:
                        continue
                    a = self.alpha(np.array([iy, ix]).T)
                    self.alpha_mat[iy, ix] = np.array([a, a, a])
        # print(len(self.alpha_mat[self.alpha_mat == 0]))
        # print(len(self.domain[self.domain == 0]) / 3)
        return self.alpha_mat
    
    def get_color_mat(self):
        if self.color_mat is None:
            self.color_mat = np.zeros(self.domain.shape, dtype=np.uint8)
            for iy in range(len(self.domain)):
                for ix in range(len(self.domain[0])):
                    if self.domain[iy, ix, 0] == 0:
                        continue
                    self.color_mat[iy, ix] = self.color(np.array([iy, ix]).T)
        return self.color_mat
    
    def is_alpha(self):
        judge = copy.deepcopy(self.get_alpha_mat())
        judge[self.domain == 0] = 1
        return np.min(judge) < 0.999