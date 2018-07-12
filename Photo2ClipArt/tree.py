import numpy as np
import copy

class Node:
    
    def __init__(self, x, used, parent=None):
        self.children = []
        self.parent = parent
        self.x = x
        self.reward = 100.0
        self.used = used
        
    def add_child(self, x, id):
        child_used = copy.deepcopy(self.used)
        child_used[id] = True
        child = Node(x, child_used, parent=self)
        self.children.append(child)
        
    def size(self):
        if len(self.children) == 0:
            return 1
        return sum([child.size() for child in self.children])
    
    def select(self):
        if len(self.children) == 0:
            return self
        self.children.sort(key=lambda x:x.reward / sum([1 if i else 0 for i in x.used]))
        # 確率が高いのは一番rewardの値が低いchildrenのはず
        return self.children[0].select()
    
    def visited_domain(self):
        ret = np.zeros(self.x[0].domain.shape, np.uint8)
        for l in self.x:
            ret += l.domain
        ret[ret != 0] = 1
        return ret
    
    def update_reward(self):
        if len(self.children) == 0:
            return self.reward
        self.reward = min([child.update_reward() for child in self.children])
        return self.reward