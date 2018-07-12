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
                key = "r:{},g:{},b:{}".format(self.segment_img[y,x,0], self.segment_img[y,x,1], self.segment_img[y,x,2])
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
        return sum([1.0 if l.is_alpha else self.beta for l in x])
    
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
        
    def fitNewLayer(self, dom):
        target = copy.deepcopy(self.input_img)
        target[dom != 1] = 0
        self.log_cnt += 1
        indx = np.where(dom == 1)
        ri = random.randint(0, len(indx[0])-1)
        pos = (indx[0][ri], indx[1][ri])
        target = np.tile(self.input_img[pos], (target.shape[0], target.shape[1], 1))
        target[dom != 1] = 0
        cv2.imwrite("old/log/log_{}.png".format(self.log_cnt), target)
        return Layer(copy.deepcopy(self.input_img[pos]), np.zeros((3,), np.uint8),\
                    np.array([1,1,1], np.uint8), np.zeros((3,), np.uint8), dom)
    
    def init_reward(self, node):
        gen_img = np.zeros(self.input_img.shape, np.uint8)
        for l in node.x:
            l_img = np.tile(l.c0, (gen_img.shape[0], gen_img.shape[1], 1))
            gen_img += l_img * l.domain + gen_img * (1 - l.domain)
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
            new_x2.append(self.fitNewLayer(copy.deepcopy(self.domains[idx])))
            base.add_child(new_x, idx)
            base.add_child(new_x2, idx)
        # reward
        for i in range(len(base.children)):
            self.init_reward(base.children[i])
        base.children.sort(key=lambda x:x.reward)
        base.children = base.children[:2]

    def monte_carlo(self):
        layer = self.fitNewLayer(copy.deepcopy(self.domains[0]))
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
       gen_layer = np.tile(res[i].c0, (res[i].domain.shape[0], res[i].domain.shape[1], 1)) # * res[i].a0
       bgr = cv2.split(gen_layer)
       bgra = cv2.merge(bgr + [res[i].domain[:,:,0] * 255])
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