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
        child = Node(self, x)
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
            return self
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

class Photo2ClipArt:

    def __init__(self, input_path, segment_path, lmd=0.5, beta=1.2):
        self.input_img = cv2.imread(input_path)
        kernel = np.ones((3,3), np.uint8)
        self.segment_img = cv2.morphologyEx(cv2.imread(segment_path), cv2.MORPH_CLOSE, kernel)
        #self.segment_img = cv2.dilate(self.segment_img, kernel, iterations=1)
        print(self.segment_img.shape)
        self.domains = []
        self.calc_domains()
        cv2.imshow('test erosion', self.segment_img)
        cv2.waitKey(0)
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
        target = self.input_img
        target[dom != 1] = 0
        # sub sampling
        indx = np.where(dom == 1)
        ri = random.randint(0, len(indx)-1) * 3
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
        cv2.imshow('Geberated Image test', gen_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
            if len(d1[d1 >= 2]) != 0:
                ret.append(i)
            elif len(d2[d2 >= 2]) != 0:
                ret.append(i)
            elif len(d3[d3 >= 2]) != 0:
                ret.append(i)
            elif len(d4[d4 >= 2]) != 0:
                ret.append(i)
        return ret

    def monte_carlo(self):
        layer = self.fitNewLayer(self.domains[0])
        root = Node([layer])
        self.init_reward(root)
        print(root.reward)
        while root.size() <= 4 * len(self.domains):
            # node selection
            base = root.select()
            # expansion
            visited = base.visited_domain()
            sindx = self.enum_regions(visited)
            
            # reward
            # back-propagation
            break

def main():
    p2c = Photo2ClipArt("img/2_cherry/input.png", "img/2_cherry/segmentation.png")
    p2c.monte_carlo()

if __name__ == '__main__':
    main()
