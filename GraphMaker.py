
import cv2
import numpy as np
import maxflow
import os
from math import *
import shutil
from WeightCalculation import *

class GraphMaker:

    foreground = 1
    background = 0

    seeds = 0
    segmented = 1

    default = 0.5
    MAXIMUM = 1000000000

    def __init__(self, gui_input_fn):
        self.image = None
        self.graph = None
        self.overlay = None
        self.seed_overlay = None
        self.segment_overlay = None
        self.mask = None
        self.background_seeds = []
        self.foreground_seeds = []
        self.background_average = np.array(3)
        self.foreground_average = np.array(3)
        self.nodes = []
        self.edges = []
        self.current_overlay = self.seeds
        self.gui_input_fn = gui_input_fn
        self.parameters = None
        self.weight_function_code = Default()

    def set_parameters(self, parameters):
        self.parameters = parameters

    def load_image(self, filename):
        filename = filename.split(',')[0][2:-1]
        self.image = cv2.imread(filename)
        if self.image is None:
            name = filename.split('/')[-1]
            shutil.copy(filename, name)
            self.image = cv2.imread(name)
            os.remove(name)
        self.graph = np.zeros_like(self.image)
        self.seed_overlay = np.zeros_like(self.image)
        self.segment_overlay = np.zeros_like(self.image)
        self.mask = None

    def add_seed(self, x, y, type):
        x = int(x)
        y = int(y)
        if y < 0 or y > self.image.shape[0] or x < 0 or x > self.image.shape[1]:
            return
        if self.image is None:
            print('Please load an image before adding seeds.')
        if type == self.background:
            if not self.background_seeds.__contains__((x, y)):
                self.background_seeds.append((x, y))
                cv2.rectangle(self.seed_overlay, (int(x)-1, int(y)-1), (int(x)+1, int(y)+1), (0, 0, 255), -1)
        elif type == self.foreground:
            if not self.foreground_seeds.__contains__((x, y)):
                self.foreground_seeds.append((x, y))
                cv2.rectangle(self.seed_overlay, (int(x)-1, int(y)-1), (int(x)+1, int(y)+1), (0, 255, 0), -1)

    def clear_seeds(self):
        self.background_seeds = []
        self.foreground_seeds = []
        self.seed_overlay = np.zeros_like(self.seed_overlay)

    def get_overlay(self):
        if self.current_overlay == self.seeds:
            return self.seed_overlay
        else:
            return self.segment_overlay

    def get_image_with_overlay(self, overlayNumber):
        if overlayNumber == self.seeds:
            return cv2.addWeighted(self.image, 0.9, self.seed_overlay, 0.4, 0.1)
        else:
            return cv2.addWeighted(self.image, 0.9, self.segment_overlay, 0.4, 0.1)

    def get_image_with_only_overlay(self):
        if self.mask is None:
            self.create_graph()
            return

        only_overlay = np.zeros_like(self.image)
        np.copyto(only_overlay, self.image, where=self.mask)
        return only_overlay

    def create_graph(self):
        if len(self.background_seeds) == 0 or len(self.foreground_seeds) == 0:
            print("Please enter at least one foreground and background seed.")
            return
        self.find_averages()
        self.populate_graph()
        self.cut_graph()

    def find_averages(self):
        self.graph = np.zeros((self.image.shape[0], self.image.shape[1]))
        self.graph.fill(self.default)
        self.background_average = np.zeros(3)
        self.foreground_average = np.zeros(3)

        for coordinate in self.background_seeds:
            self.graph[coordinate[1] - 1, coordinate[0] - 1] = 0

        for coordinate in self.foreground_seeds:
            self.graph[coordinate[1] - 1, coordinate[0] - 1] = 1

    def __neighbor_weight_function(self, index, neighbor):
        return self.weight_function_code.get_weight(self.image, index, neighbor, self.parameters, gui_input_fn=self.gui_input_fn)

    def populate_graph(self):
        self.nodes = []
        self.edges = []

        for (y, x), value in np.ndenumerate(self.graph):
            if value == 0.0:
                self.nodes.append((self.get_node_num(x, y, self.image.shape), self.MAXIMUM, 0))

            elif value == 1.0:
                self.nodes.append((self.get_node_num(x, y, self.image.shape), 0, self.MAXIMUM))

            else:
                self.nodes.append((self.get_node_num(x, y, self.image.shape), 0, 0))

        for (y, x), value in np.ndenumerate(self.graph):
            if y == self.graph.shape[0] - 1 or x == self.graph.shape[1] - 1:
                continue
            my_index = self.get_node_num(x, y, self.image.shape)

            neighbor_index = self.get_node_num(x+1, y, self.image.shape)
            g = self.__neighbor_weight_function((x, y), (x+1, y))
            self.edges.append((my_index, neighbor_index, g))

            neighbor_index = self.get_node_num(x, y+1, self.image.shape)
            g = self.__neighbor_weight_function((x, y), (x, y+1))
            self.edges.append((my_index, neighbor_index, g))

    def cut_graph(self):
        self.segment_overlay = np.zeros_like(self.segment_overlay)
        self.mask = np.zeros_like(self.image, dtype=bool)
        g = maxflow.Graph[float](len(self.nodes), len(self.edges))
        nodelist = g.add_nodes(len(self.nodes))

        for node in self.nodes:
            g.add_tedge(nodelist[node[0]], node[1], node[2])

        for edge in self.edges:
            g.add_edge(edge[0], edge[1], edge[2], edge[2])

        flow = g.maxflow()

        for index in range(len(self.nodes)):
            if g.get_segment(index) == 1:
                xy = self.get_xy(index, self.image.shape)
                self.segment_overlay[int(xy[1]), int(xy[0])] = (255, 0, 255)
                self.mask[int(xy[1]), int(xy[0])] = (True, True, True)

    def swap_overlay(self, overlay_num):
        self.current_overlay = overlay_num

    def save_image(self, filename):
        if self.mask is None:
            print('Please segment the image before saving.')
            return

        to_save = np.zeros_like(self.image)
        np.copyto(to_save, self.image, where=self.mask)
        cv2.imwrite(str(filename), to_save)

    @staticmethod
    def get_node_num(x, y, array_shape):
        return y * array_shape[1] + x

    @staticmethod
    def get_xy(nodenum, array_shape):
        return (nodenum % array_shape[1]), (nodenum / array_shape[1])
