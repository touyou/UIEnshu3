import cv2
import numpy as np
import random
import sys
import os
import copy
from PIL import Image
from xml.dom import minidom

from tree import *
from layer import *
from helper import *

class Photo2ClipArt:
    
    def __init__(self, input_path, segment_path, lmd=0.5, beta=1.2):
        print(input_path)
        print(segment_path)
        self.input_img = cv2.imread(input_path)
        kernel = np.ones((3, 3), np.uint8)
        self.segment_img = cv2.morphologyEx(cv2.imread(segment_path), cv2.MORPH_CLOSE, kernel)
        self.domains = []
        self.calc_domains()
        print("segmentation number: {}".format(len(self.domains)))
        self.lmd = lmd
        self.beta = beta
        self.log_cnt = 0
        
    def calc_domains(self):
        domain_dict = {}
        width = self.segment_img.shape[1]
        height = self.segment_img.shape[0]
        for y in range(height):
            for x in range(width):
                key = "b:{},g:{},r:{}".format(self.segment_img[y,x,0], self.segment_img[y,x,1], self.segment_img[y,x,2])
                if key in domain_dict:
                    self.domains[domain_dict[key]][y,x] = 1
                else:
                    domain_dict[key] = len(self.domains)
                    new_mask = np.zeros(self.segment_img.shape, np.uint8)
                    new_mask[y,x] = 1
                    self.domains.append(new_mask)
        # for d in self.domains:
        #     print(len(d[d == 1]))
        print("domain dict: {}".format(domain_dict))
    
    def simplicity(self, x):
        N = len(x)
        return sum([1.0 if l.is_alpha() else self.beta for l in x])
    
    def fidelity(self, x, new_img):
        domain = np.zeros(new_img.shape, np.uint8)
        for l in x:
            domain += l.domain
        domain[domain != 0] = 1
        origin = self.input_img * domain
        new = new_img * domain
        # print("origin: {}, new: {}, dom: {}".format(len(origin[origin != 0]), len(new[new != 0]), np.sum(domain)))
        return np.sum((origin - new) ** 2) / np.sum(domain)
    
    def energy(self, x, new_img):
        # print("fidelity: {}, simplicity: {}".format(self.fidelity(x, new_img), self.simplicity(x)))
        return (1.0 - self.lmd) * self.fidelity(x, new_img) + self.lmd * self.simplicity(x)

    def jacobian_line(self, iy, ix, xs, sub_img):
        def jacobi_c0(img_x, c0, c1, oc, p, a0, a1, oa, sub_img_x): # shape (3,)
            return 2 * (img_x - (c0 + c1 * oc.dot(p)) * (a0 + a1 * oa.dot(p)) - (1 - a0 - a1 * oa.dot(p)) * sub_img_x) * (-(a0 + a1 * oa.dot(p)))
        def jacobi_c1(img_x, c0, c1, oc, p, a0, a1, oa, sub_img_x): # shape (3,)
            return 2 * (img_x - (c0 + c1 * oc.dot(p)) * (a0 + a1 * oa.dot(p)) - (1 - a0 - a1 * oa.dot(p)) * sub_img_x) * (-(a0 + a1 * oa.dot(p))) * oc.dot(p)
        def jacobi_a0(img_x, c0, c1, oc, p, a0, a1, oa, sub_img_x): # shape (3,)のsumがすべての微分
            return 2 * (img_x - (c0 + c1 * oc.dot(p)) * (a0 + a1 * oa.dot(p)) - (1 - a0 - a1 * oa.dot(p)) * sub_img_x) * (sub_img_x - (c0 + c1 * oc.dot(p)))
        def jacobi_a1(img_x, c0, c1, oc, p, a0, a1, oa, sub_img_x): # 同上
            return 2 * (img_x - (c0 + c1 * oc.dot(p)) * (a0 + a1 * oa.dot(p)) - (1 - a0 - a1 * oa.dot(p)) * sub_img_x) * (sub_img_x - (c0 + c1 * oc.dot(p)))  * oa.dot(p)
        def jacobi_oc(img_x, c0, c1, oc, p, a0, a1, oa, sub_img_x, idx):
            return 2 * (img_x - (c0 + c1 * oc.dot(p)) * (a0 + a1 * oa.dot(p)) - (1 - a0 - a1 * oa.dot(p)) * sub_img_x) * (- c1 * p[idx,0] * (a0 + a1 * oa.dot(p)))
        def jacobi_oa(img_x, c0, c1, oc, p, a0, a1, oa, sub_img_x, idx):
            return 2 * (img_x - (c0 + c1 * oc.dot(p)) * (a0 + a1 * oa.dot(p)) - (1 - a0 - a1 * oa.dot(p)) * sub_img_x) * (a1 * p[idx, 0] * sub_img_x - a1 * p[idx, 0] * (c0 + c1 * oc.dot(p)))
        img = copy.deepcopy(self.input_img[iy, ix])
        c0 = np.array([xs[0], xs[1], xs[2]], dtype=np.uint8)
        c1 = np.array([xs[3], xs[4], xs[5]], dtype=np.uint8)
        a0 = xs[6]
        a1 = xs[7]
        oc = np.array([xs[8], xs[9]])
        oa = np.array([xs[10], xs[11]])
        p = np.array([iy, ix]).T
        
        jc0 = jacobi_c0(img, c0, c1, oc, p, a0, a1, oa, sub_img)
        jc1 = jacobi_c1(img, c0, c1, oc, p, a0, a1, oa, sub_img)
        ja0 = np.sum(jacobi_a0(img, c0, c1, oc, p, a0, a1, oa, sub_img))
        ja1 = np.sum(jacobi_a1(img, c0, c1, oc, p, a0, a1, oa, sub_img))
        joc0 = np.sum(jacobi_oc(img, c0, c1, oc, p, a0, a1, oa, sub_img, 0))
        joc1 = np.sum(jacobi_oc(img, c0, c1, oc, p, a0, a1, oa, sub_img, 1))
        joa0 = np.sum(jacobi_oa(img, c0, c1, oc, p, a0, a1, oa, sub_img, 0))
        joa1 = np.sum(jacobi_oa(img, c0, c1, oc, p, a0, a1, oa, sub_img, 1))
        return [jc0[0], jc0[1], jc0[2], jc1[0], jc1[1], jc1[2], ja0, ja1, joc0, joc1, joa0, joa1]
    
    def fit_image(self, iy, ix, xs, sub_img):
        return (self.input_img[iy,ix,0] - (xs[0] + xs[3] * (xs[8] * iy + xs[9] * ix)) * (xs[6] + xs[7] * (xs[10] * iy + xs[11] * ix)) + (1 - (xs[6] + xs[7] * (xs[10] * iy + xs[11] * ix))) * sub_img[0]) ** 2 \
            + (self.input_img[iy,ix,1] - (xs[1] + xs[4] * (xs[8] * iy + xs[9] * ix)) * (xs[6] + xs[7] * (xs[10] * iy + xs[11] * ix)) + (1 - (xs[6] + xs[7] * (xs[10] * iy + xs[11] * ix))) * sub_img[1]) ** 2 \
            + (self.input_img[iy,ix,2] - (xs[2] + xs[5] * (xs[8] * iy + xs[9] * ix)) * (xs[6] + xs[7] * (xs[10] * iy + xs[11] * ix)) + (1 - (xs[6] + xs[7] * (xs[10] * iy + xs[11] * ix))) * sub_img[2]) ** 2
    
    def fitNewLayer(self, dom, x):
        # subsampling
        target = copy.deepcopy(self.input_img)
        target[dom != 1] = 0
        idx = np.where(dom == 1)
        # 12pixel以下は別のレイヤーの一部にしたいけど一旦適当に単色にしておく方針で
        if len(idx) < 12:
            return Layer(copy.deepcopy(self.input_img[idx[0][0], idx[1][0]]), np.zeros((3,), np.uint8), 1.0, 0.0, np.array([0, 0]), np.array([0, 0]), dom)
        sub_idx = random.sample(idx, 12)
        sub_pos = [(idx[0][ri], idx[1][ri]) for ri in sub_idx]
        sub_img = []
        for (iy, ix) in sub_pos:
            img = np.array([255, 255, 255], dtype=np.uint8)
            p = np.array([iy, ix]).T
            for layer in x:
                img += layer.color(p) * layer.alpha(p) + (1 - layer.alpha(p)) * img
            sub_img.append(img)
        
        # Newton-Raphson
        w = self.input_img.shape[1]
        h = self.input_img.shape[0]
        ## 初期値
        c0 = copy.deepcopy(self.input_img[sub_pos[6]])
        c1 = copy.deepcopy(self.input_img[sub_pos[6]])
        a0 = 1.0
        a1 = 1.0
        oc = np.array([0, 0])
        oa = np.array([0, 0])
        # x = np.array([c0[0], c0[1], c0[2], c1[0], c1[1], c1[2], a0, a1, oc[0], oc[1], oa[0], oa[1]])
        # for _ in range(5000):
        #     J = []
        #     f = []
        #     for i in range(12):
        #         J.append(self.jacobian_line(sub_pos[i][0], sub_pos[i][1], x, sub_img[i]))
        #         f.append(self.fit_image(sub_pos[i][0], sub_pos[i][1], x, sub_img[i]))
        #     J = np.array(J)
        #     f = np.array(f)
        #     x -= np.linalg.inv(J).dot(f)
        # c0 = np.array([x[0], x[1], x[2]], dtype=np.uint8)
        # c1 = np.array([x[3], x[4], x[5]], dtype=np.uint8)
        # a0 = x[6]
        # a1 = x[7]
        # oc = np.array([x[8], x[9]])
        # oa = np.array([x[10], x[11]])
        # target = np.tile(self.input_img[pos], (target.shape[0], target.shape[1], 1))
        # target[dom != 1] = 0
        # cv2.imwrite("old/log/log_{}.png".format(self.log_cnt), target)
        return Layer(c0, c1, a0, a1, oc, oa, dom)
    
    def init_reward(self, node):
        gen_img = np.zeros(self.input_img.shape, np.uint8)
        for l in node.x:
            # l_img = np.tile(l.c0, (gen_img.shape[0], gen_img.shape[1], 1))
            # gen_img += l_img * l.domain + gen_img * (1 - l.domain)
            gen_img += (l.get_color_mat().astype(np.float64) * l.get_alpha_mat() + (1 - l.get_alpha_mat()) * gen_img.astype(np.float64)).astype(np.uint8)
        node.reward = self.energy(node.x, gen_img)
    
    def gen_child(self, base):
        # expansion
        # visited = base.visited_domain()
        sindx = []
        for i in range(len(self.domains)):
            if not base.used[i]:
                sindx.append(i)
        if len(sindx) == 0:
            return
        # print(len(base.x[0].domain[base.x[0].domain == 1]))
        for idx in sindx:
            extend_layer = 0
            for j in range(len(base.x)):
                if is_adj(self.domains[idx], base.x[j].domain):
                    extend_layer = j
                    break
            # extend existing layer
            new_x = copy.deepcopy(base.x)
            new_x[extend_layer].domain += self.domains[idx]
            new_x[extend_layer].domain[new_x[extend_layer].domain != 0] = 1
            # generate new layer
            new_x2 = copy.deepcopy(base.x)
            new_x2.append(self.fitNewLayer(copy.deepcopy(self.domains[idx]), copy.deepcopy(base.x)))
            base.add_child(new_x, idx)
            base.add_child(new_x2, idx)
        # reward
        for i in range(len(base.children)):
            self.init_reward(base.children[i])
        base.children.sort(key=lambda x:x.reward)
        base.children = base.children[:2]

    def monte_carlo(self):
        layer = self.fitNewLayer(copy.deepcopy(self.domains[0]), [])
        used = [False for _ in range(len(self.domains))]
        used[0] = True
        root = Node([layer], used=used)
        self.init_reward(root)
        print("root domain: {}".format(len(root.x[0].domain[root.x[0].domain == 1])))
        print("first root reward = {}".format(root.reward))
        iter_cnt = 0
        while root.size() <= 4 * len(self.domains):
            # node selection
            base = root.select()
            # self.gen_child(base)
            if base.parent is None:
                self.gen_child(base)
            else:
                for i in range(len(base.parent.children)):
                    self.gen_child(base.parent.children[i])
            # back propagation
            root.update_reward()
            iter_cnt += 1
            print("root domain: {}".format(len(root.x[0].domain[root.x[0].domain == 1])))
            print("{}: size {}, root ene: {}".format(iter_cnt, root.size(), root.reward))
        return root


def main(file_name):
    p2c = Photo2ClipArt("img/{}/input.png".format(file_name), "img/{}/segmentation.png".format(file_name))
    root = p2c.monte_carlo()
    images(root, "0", file_name)
    res = root.select().x
    for i in range(len(res)):
    #    gen_layer = np.tile(res[i].c0, (res[i].domain.shape[0], res[i].domain.shape[1], 1)) # * res[i].a0
        gen_layer = (res[i].get_color_mat().astype(np.float64) * res[i].get_alpha_mat()).astype(np.uint8)
        bgr = cv2.split(gen_layer)
        bgra = cv2.merge(bgr + res[i].get_alpha_mat()[:,:,0] * 255)
        bgra[res[i].domain == 0] = 0
        cv2.imwrite("old/res/layer_{}.png".format(i), bgra)
        path_layer = np.zeros(res[i].domain.shape, np.uint8)
        path_layer[res[i].domain == 0] = 255
        cv2.imwrite("old/res/temp_{}.ppm".format(i), path_layer)
        os.system('potrace old/res/temp_{}.ppm -s -o old/res/temp_{}.svg'.format(i, i))
    xdocs = []
    for i in range(len(res)):
        xdocs.append(minidom.parse("old/res/temp_{}.svg".format(i)))
    os.system("rm old/res/*.ppm")

if __name__ == '__main__':
    main(sys.argv[1])