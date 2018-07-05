import cv2
import numpy as np
import random
from hashlib import sha1


class Layer:

    def __init__(self, c0, c1, a0, a1, domain):
        self.c0 = c0
        self.c1 = c1
        self.a0 = a0
        self.a1 = a1
        self.domain = domain
        self.is_alpha = (a0 < 0.999).all()

    def alpha(self, p):
        return self.a0 + self.a1 * self.o.dot(p)

    def color(self, p):
        return self.c0 + self.c1 * self.o.dot(p)

class Node:

    def __init__(self, x, parent=None):
        self.children = []
        self.parent = parent
        self.x = x
        self.reward = 0.0

    def add_child(self, x):
        child = Node(x, parent=self)
        self.children.append(child)

    def size(self):
        if len(self.children) == 0:
            return 1
        ret = 0
        for child in self.children:
            ret += child.size()
        return ret

    def select(self):
        if len(self.children) == 0:
            return self.parent
        # sort children by reward and select one
        self.children.sort(key=lambda x:x.reward)
        if len(self.children) < 2:
            return self.children[0].select()
        a_exp = np.exp(-self.children[0].reward)
        b_exp = np.exp(-self.children[1].reward)
        if a_exp < b_exp:
            return self.children[1].select()
        else:
            return self.children[0].select()

    def visited_domain(self):
        ret = np.zeros(self.x[0].domain.shape, np.uint8)
        for l in self.x:
            ret += l.domain
        return ret

    def update_reward(self):
        if len(self.children) == 0:
            return self.reward
        min_reward = 1000000
        for child in self.children:
            min_reward = min(min_reward, child.update_reward())
        self.reward = min_reward
        return self.reward

class Photo2ClipArt:

    def __init__(self, input_path, segment_path, lmd=0.5, beta=1.2):
        self.input_img = cv2.imread(input_path)
        kernel = np.ones((3,3), np.uint8)
        self.segment_img = cv2.morphologyEx(cv2.imread(segment_path), cv2.MORPH_CLOSE, kernel)
        #self.segment_img = cv2.dilate(self.segment_img, kernel, iterations=1)
        print(self.segment_img.shape)
        self.domains = []
        self.calc_domains()
        #cv2.imshow('test erosion', self.input_img)
        #cv2.waitKey(0)
        #dst = self.input_img
        #dst[self.domains[6] != 1] = 255
        #cv2.imshow('layer1', dst)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        print(len(self.domains))
        self.lmd = lmd
        self.beta = beta

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
        print(domain_dict)

    def simplicity(self, x):
        N = len(x)
        return sum([1.0 if l.is_alpha else self.beta for l in x]) / N

    def fidelity(self, x, new_img):
        domain = np.zeros(new_img.shape, np.uint8)
        for l in x:
            domain += l.domain
        domain[domain != 0] = 1
        origin = self.input_img * domain
        new = new_img * domain
        return np.sum((origin - new) ** 2) / np.sum(domain)

    def energy(self, x, new_img):
        return (1.0 - self.lmd) * self.fidelity(x, new_img) + self.lmd * self.simplicity(x)

    def fitNewLayer(self, dom):
        target = self.input_img.copy()
        target[dom != 1] = 0
        # sub sampling
        indx = np.where(dom == 1)
        ri = random.randint(0, len(indx[0])-1)
        pos = (indx[0][ri], indx[1][ri])
        return Layer(self.input_img[pos], np.zeros((3,),np.uint8),\
                np.array([1,1,1], dtype=np.uint8), np.zeros((3,),np.uint8), dom)
        #for _ in range(10):
        #    rx = random.randint(0, num-1) * 3
        #    pos1 = (indx[0][rx], indx[1][rx])
        #    for _ in range(10):
        #        ry = random.randint(0, num-1) * 3
        #        pos2 = (indx[0][ry], indx[1][ry])

    def init_reward(self, node):
        gen_img = np.zeros(self.input_img.shape, np.uint8)
        for l in node.x:
            gen_img += l.domain * l.c0 * l.a0 + (1 - l.domain * l.a0) * gen_img

        node.reward = self.energy(node.x, gen_img)
        #cv2.imshow('Geberated Image test', gen_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    def enum_regions(self, visited):
        ret = []
        for (i, domain) in enumerate(self.domains):
            check_vis = visited + domain
            if len(check_vis[check_vis >= 2]) != 0:
                continue
            d1 = np.roll(domain, 1, axis=0)
            d1[:,0] = 0
            d1 += visited
            d2 = np.roll(domain, -1, axis=0)
            d2[:,-1] = 0
            d2 += visited
            d3 = np.roll(domain, 1, axis=1)
            d3[0] = 0
            d3 += visited
            d4 = np.roll(domain, -1, axis=1)
            d4[-1] = 0
            d4 += visited
            if len(d1[d1 >= 2]) != 0 or len(d2[d2 >= 2]) != 0 or len(d3[d3 >= 2]) != 0 or len(d4[d4 >= 2]) != 0:
                ret.append(i)
            elif i == len(self.domains)-1:
                ret.append(i)
        return ret

    def gen_child(self, base):
        # expansion
        visited = base.visited_domain()
        sindx = self.enum_regions(visited)
        if len(sindx) == 0:
            return
        for idx in sindx:
            extend_layer = 0
            for j in range(len(base.x)):
                d1 = np.roll(self.domains[idx], 1, axis=0)
                d1[:,0] = 0
                d1 += base.x[j].domain
                d2 = np.roll(self.domains[idx], -1, axis=0)
                d2[:,-1] = 0
                d2 += base.x[j].domain
                d3 = np.roll(self.domains[idx], 1, axis=1)
                d3[0] = 0
                d3 += base.x[j].domain
                d4 = np.roll(self.domains[idx], -1, axis=1)
                d4[-1] = 0
                d4 += base.x[j].domain
                if len(d1[d1 >= 2]) != 0 or len(d2[d2 >= 2]) != 0 or len(d3[d3 >= 2]) != 0 or len(d4[d4 >= 2]) != 0:
                    extend_layer = j
                    break
            # extend existing layer
            new_x = base.x.copy()
            new_x[extend_layer].domain += self.domains[idx]
            # generate new layer
            new_x2 = base.x.copy()
            new_x2.append(self.fitNewLayer(self.domains[idx].copy()))
            base.add_child(new_x)
            base.add_child(new_x2)
        # reward
        for i in range(len(base.children)):
            self.init_reward(base.children[i])

    def monte_carlo(self):
        layer = self.fitNewLayer(self.domains[0].copy())
        root = Node([layer])
        self.init_reward(root)
        print(root.reward)
        iter_count = 0
        pre_size = 0
        while root.size() <= 4 * len(self.domains) and pre_size != root.size():
            pre_size = root.size()
            # node selection
            base_p = root.select()
            if base_p is None:
                self.gen_child(root)
            else:
                self.gen_child(base_p.children[0])
                self.gen_child(base_p.children[1])
            # back-propagation
            root.update_reward()
            iter_count += 1
            print("{}: size {}, root ene: {}".format(iter_count, root.size(), root.reward))
        return root

def images(node, idx):
    gen_img = np.zeros(node.x[0].domain.shape, np.uint8)
    for l in node.x:
        gen_img += l.domain * l.c0 * l.a0 + (1 - l.domain * l.a0) * gen_img
    bgr = cv2.split(gen_img)
    bgra = cv2.merge(bgr + [node.visited_domain()[:,:,0] * 255])
    cv2.imwrite("res/result_{}.png".format(idx), bgra)
    for i in range(len(node.children)):
        images(node.children[i], idx + str(i))

def main():
    p2c = Photo2ClipArt("img/9_egg/input.png", "img/9_egg/segmentation.png")
    root = p2c.monte_carlo()
    images(root, "0")
    cv2.imshow('input', p2c.input_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #for i in range(len(res)):
    #    gen_layer = res[i].domain * res[i].c0 * res[i].a0
    #    bgr = cv2.split(gen_layer)
    #    bgra = cv2.merge(bgr + [res[i].domain[:,:,0] * 255])
    #    cv2.imwrite("res/layer_{}.png".format(i), bgra)

if __name__ == '__main__':
    main()
