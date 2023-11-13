from .WeightCalc import WeightCalc
from math import exp
import numpy as np

class Default(WeightCalc):
    def __init__(self):
        self.name = "Default"
    
    def get_name(self):
        return self.name

    def get_weight(self, image, index, neighbor, parameters, **kwargs):
        if parameters is None:
            parameters = kwargs['gui_input_fn']()
        x1, y1 = index
        x2, y2 = neighbor
        k = parameters['K']
        s = parameters['s']
        return k*exp(-(abs(np.sum(image[y1, x1]) - np.sum(image[y2, x2]))**2)/s)
