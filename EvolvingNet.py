import random

import math
import numpy

from MatrixNet import MatrixNet
from formulas import sigmoid, sigmoid_der

def color_formula(x):
    return [int(x * 255.), 255, int(x * 255.)]


class EvolvingNet(MatrixNet):
    def __init__(self, in_dem, out_dem, genetics_nodes, genetics_weights, mutability=.5, activation=sigmoid,
                 activation_der=sigmoid_der, color_formula_param=color_formula):
        self.InDem = in_dem
        self.OutDem = out_dem
        self.Mutability = mutability
        self.Genetics = (genetics_nodes, genetics_weights)
        Dem = []
        Dem.append(in_dem)
        for layer in range(len(genetics_nodes)):
            Dem.append(len(genetics_nodes[layer]))
        Dem.append(out_dem)
        MatrixNet.__init__(self, Dem, [0.0, 0.0], activation, activation_der, color_formula_param)

        for layer in range(len(genetics_nodes)):
            if len(genetics_weights) > layer:
                for node in range(len(genetics_nodes[layer])):
                    if len(genetics_weights[layer]) > genetics_nodes[layer][node]:
                        for source, weight in genetics_weights[layer][genetics_nodes[layer][node]]:
                            self.WeightArray[layer + 1][node][source] = weight
        if len(genetics_weights) > len(genetics_nodes):
            layer = -1
            for node in range(out_dem):
                if len(genetics_weights[layer]) > node:
                    for source, weight in genetics_weights[layer][node]:
                        self.WeightArray[layer + 1][node][source] = weight

    def compatable(self, evolvingNet2):
        return self.InDem == evolvingNet2.InDem and self.OutDem == evolvingNet2.OutDem

    def breed(self, evolvingNet2):
        assert self.compatable(evolvingNet2)

        genetics_nodes = self.Genetics[0]
        genetics_weights = self.Genetics[1]

        for layer in range(len(genetics_nodes)):
            for node in evolvingNet2.Genetics[0][layer]:
                if node not in genetics_nodes[layer]:
                    genetics_nodes[layer].append(node)

        for layer in range(len(genetics_weights)):
            for node in range(len(genetics_weights[layer])):
                if len(evolvingNet2.Genetics[1][layer]) > node:
                    for weight in evolvingNet2.Genetics[1][layer][node]:
                        contain_check = False
                        for weight_t in self.Genetics[1][layer][node]:
                            if weight[0] == weight_t[0]:
                                contain_check = True
                                break
                        if not contain_check:
                            genetics_weights[layer][node].append(weight)
        nodes_to_add = []

        for layer in range(len(genetics_weights)):
            nodes_to_add.append([])
            for node in range(len(genetics_weights[layer])):
                removables = []
                sources = []
                if len(genetics_weights[layer][node]) > 0 and layer < len(genetics_nodes) and node not in genetics_nodes[layer]:
                    nodes_to_add[layer].append(node)
                for weight in range(len(genetics_weights[layer][node])):
                    genetics_weights[layer][node][weight] = (genetics_weights[layer][node][weight][0],
                                                             self.mutate(genetics_weights[layer][node][weight][1]))
                    sources.append(genetics_weights[layer][node][weight][0])
                    if random.random() < self.Mutability:
                        removables.append(genetics_weights[layer][node][weight])
                for weight in removables:
                    genetics_weights[layer][node].remove(weight)
                if layer - 1 >= 0:
                    for source in range(len(genetics_nodes[layer - 1])):
                        if source not in sources and random.random() < self.Mutability:
                            genetics_weights[layer][node].append((source, -1.0 + self.mutate(1.0)))
                else:
                    for source in range(self.InDem):
                        if source not in sources and random.random() < self.Mutability:
                            genetics_weights[layer][node].append((source, -1.0 + self.mutate(1.0)))

        for layer in range(len(genetics_nodes)):
            n = 0
            while n < len(genetics_nodes[layer]):
                if random.random() < self.Mutability:
                    del genetics_nodes[layer][n]
                else:
                    n += 1
            for node in nodes_to_add[layer]:
                if random.random() < self.Mutability:
                    genetics_nodes[layer].append(node)

        newNet = EvolvingNet(self.InDem, self.OutDem,
                             genetics_nodes, genetics_weights,
                             mutability=self.Mutability)

        return newNet

    def mutate(self, number):
        return number + (0.5 - random.random()) * 2.0 * self.Mutability


    def distance(self, net):
        diff = abs(len(self.Genetics[0]) - len(net.Genetics[0])) * 100
        for layer in range(max(len(self.Genetics[0]), len(net.Genetics[0]))):
            if len(self.Genetics[0]) > layer and len(net.Genetics[0]) > layer:
                diff += abs(len(self.Genetics[0][layer]) - len(net.Genetics[0][layer]))
            elif len(self.Genetics[0]) > layer:
                diff += len(self.Genetics[0][layer])
            elif len(net.Genetics[0]) > layer:
                diff += len(net.Genetics[0][layer])
        diff *= 10000
        for layer in range(max(len(self.Genetics[1]), len(net.Genetics[1]))):
            if len(self.Genetics[1]) > layer and len(net.Genetics[1]) > layer:
                for node in range(max(len(self.Genetics[1][layer]), len(net.Genetics[1][layer]))):
                    if len(self.Genetics[1][layer]) > node and len(net.Genetics[1][layer]) > node:
                        self_list = self.Genetics[1][layer][node]
                        net_list = net.Genetics[1][layer][node]
                        w1 = 0
                        while w1 < len(self_list):
                            w2 = 0
                            remove = False
                            while w2 < len(net_list) and not remove:
                                if self_list[w1][0] == net_list[w2][0]:
                                    diff += int(abs(self_list[w1][1] - net_list[w2][1]) * 10)
                                    del self_list[w1][0]
                                    del net_list[w2][0]
                                    remove = True
                                else:
                                    w2 += 1
                            if not remove:
                                w1 += 1
                        diff += 100 * (len(self_list) + len(net_list))
        return diff