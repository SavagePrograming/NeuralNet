import random
from typing import Callable, List

import numpy

from Nets.MatrixNet import MatrixNet
from formulas import distance_formula, sigmoid, sigmoid_der, color_formula


class StaticEvolvingNet(MatrixNet):
    def __init__(self,
                 in_dem: int,
                 out_dem: int,
                 genetics_layers: int,
                 genetics_weights: List[numpy.array] = None,
                 mutability: float = .5,
                 activation: Callable = sigmoid,
                 activation_der: Callable = sigmoid_der,
                 color_formula_param: Callable = color_formula):
        self.in_dem: int = in_dem
        self.out_dem: int = out_dem
        self.mutability: float = mutability
        self.layers: int = genetics_layers

        dem = []
        dem.append(in_dem)
        for layer in range(genetics_layers):
            dem.append(max(in_dem, out_dem))
        dem.append(out_dem)

        MatrixNet.__init__(self, dem, [self.mutability, -self.mutability], activation, activation_der,
                           color_formula_param)

        if genetics_weights is not None:
            self.weight_array = genetics_weights

    def compatible(self, net):
        return self.in_dem == net.InDem and \
               self.out_dem == net.OutDem and \
               self.layers == net.Layers

    def breed(self, net):
        assert self.compatible(net)
        newWeights = []
        for i in range(len(self.weight_array)):
            newWeights.append([])
            for j in range(len(self.weight_array[i])):
                newWeights[i].append([])
                for k in range(len(self.weight_array[i][j])):
                    newWeights[i][j].append(random.choice([self.weight_array[i][j][k],
                                                           net.WeightArray[i][j][k]]))
            newWeights[i] = numpy.array(newWeights[i])

        newNet = StaticEvolvingNet(self.in_dem, self.out_dem, self.layers, genetics_weights=newWeights)
        return newNet

    def replicate(self):
        newWeights = []
        for i in range(len(self.weight_array)):
            newWeights.append([])
            for j in range(len(self.weight_array[i])):
                newWeights[i].append([])
                for k in range(len(self.weight_array[i][j])):
                    newWeights[i][j].append(self.mutate(self.weight_array[i][j][k]))
            newWeights[i] = numpy.array(newWeights[i])

        newNet = StaticEvolvingNet(self.in_dem, self.out_dem, self.layers, genetics_weights=newWeights)
        return newNet

    def mutate(self, number):
        return number + (0.5 - random.random()) * 2.0 * self.mutability

    def distance(self, net):
        return sum(map(distance_formula, self.weight_array, net.WeightArray))
