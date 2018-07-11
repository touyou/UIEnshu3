import cv2
import numpy as numpy
import random

class Photo2ClipArt:
    
    def __init__(self, input_path, segment_path, lmd=0.5, beta=1.2):
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
        return np.sum((origin - new) ** 2) / np.sum(domain)
    
    def energy(self, x, new_img):
        return (1.0 - self.lmd) * self.fidelity(x, new_img) + self.lmd * self.simplicity(x)
        
    def fitNewLayer(self, dom):
        target = self.input_img.copy()
        target[dom != 1] = 0
        self.log_cnt += 1
        indx = np.where(dom == 1)
        ri = random.randint(0, len(indx[0])-1)
        pos = (indx[0][ri], indx[1][ri])
        target[dom == np.array([1,1,1])] = self.input_img[pos].copy()
        cv2.imwrite("../old/log/log_{}.png".format(self.log_cnt), target)
        return Layer(self.input_img[pos].copy(), npl.zeros((3,), np.uint8),\
                    np.array([1,1,1], np.uint8), np.zeros((3,), np.uint8), dom)
    
    def init_reward(self, node):
        gen_img = np.zeros(self.input_img.shape, np.uint8)
        for l in node.x:
            gen_img += l.domain * l.c0 + (1 - l.domain) * gen_img
        node.reward = self.energy(node.x, gen_img)
    
    def gen_child(self, base):
        # expansion
        visited = base.visited_domain()
        sindx = []
        for i in range(len(self.domains)):
            if not base.used[i]:
                sindx.append(i)
        if len(sindx) == 0:
            return
        for idx in sindx:
            extend_layer = 0
            for j in range(len(base.x)):
                if is_adj(self.domains[idx], base.x[j].domain):
                    extend_layer = j
                    break
            # extend existing layer
            new_x = base.x.copy()
            new_x[extend_layer].domain += self.domains[idx]
            


def main(file_name):
    pass

if __name__ == '__main__':
    main(sys.argv[1])