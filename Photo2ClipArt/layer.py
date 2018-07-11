import numpy as np

class Layer:
    
    def __init__(self, c0, c1, a0, a1, domain):
        self.c0 = c0
        self.c1 = c1
        self.a0 = a0
        self.a1 = a1
        self.domain = domain
        self.is_alpha = (a0 < 0.999).all()
