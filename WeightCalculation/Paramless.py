import numpy as np
from .WeightCalc import WeightCalc

class Paramless(WeightCalc):
    def __init__(self):
        self.name = "Paramless"
    
    def get_name(self):
        return self.name
    
    def get_weight(self, image, index, neighbor, parameters, **kwargs):
        x1, y1 = index
        x2, y2 = neighbor
        return 1 / (1 + np.sum(np.power(image[y1, x1] - image[y2, x2], 2)))

