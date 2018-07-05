import numpy as np

'''
Union Find Tree
http://www.geocities.jp/m_hiroi/light/pyalgo61.html
https://github.com/aratakokubun/GraphBasedSegmentationWithUnionFind/blob/master/UnionFind.py
'''
class UnionFind:

    def __init__(self, size):
        self.table = [(-1, id) for id in range(size)]

    def find(self, x):
        p_id = self.table[x][1]
        while self.table[x][0] >= 0:
            p_id = self.table[x][1]
            x = self.table[x][0]
        return (x, p_id)

    def union(self, x, y):
        (s1, _) = self.find(x)
        (s2, _) = self.find(y)
        if s1 != s2:
            if self.table[s1][0] >= self.table[s2][0]:
                self.table[s1] = (self.table[s1][0] + self.table[s2][0], self.table[s1][1])
                self.table[s2] = (s1, self.table[s2][1])
            else:
                self.table[s2] = (self.table[s1][0] + self.table[s2][0], self.table[s2][1])
                self.table[s1] = (s2, self.table[s1][1])
            return True
        return False

'''
Wrapper Union Find for Image pixels
'''
class UnionFindImage:

    def __init__(self, pixels):
        self.height = pixels.shape[0]
        self.width = pixels.shape[1]
        self.make_tree(pixels)

    def make_tree(self, pixels):
        self.tree = UnionFind(self.width * self.height)
        self.tree_dict[pixels[0, 0]] = 0
        for row in range(self.height):
            for col in range(self.width):
                if row == 0 and col == 0:
                    continue
                if self.tree_dict[pixels[row, col]] is None:
                    self.tree_dict[pixels[row, col]] = self.convert_pixel(col, row)
                else:
                    self.tree.union(self.tree_dict[pixels[row, col]], self.convert_pixel(col, row))
                    self.tree_dict[pixels[row, col]] = self.tree.find(self.convert_pixel(col, row))[1]

    def convert_id(self, id):
        return (id / self.width, id % self.width)

    def convert_pixel(self, x, y):
        return y * self.width + x
