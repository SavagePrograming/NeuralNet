import random

import numpy

from Nets.MatrixNet import MatrixNet
from formulas import distance_formula, sigmoid, sigmoid_der, color_formula


class StaticEvolvingNet(MatrixNet):
    def __init__(self, in_dem, out_dem, genetics_layers, genetics_weights=None, mutability=.5, activation=sigmoid,
                 activation_der=sigmoid_der, color_formula_param=color_formula):
        self.InDem = in_dem
        self.OutDem = out_dem
        self.Mutability = mutability
        self.Layers = genetics_layers

        Dem = []
        Dem.append(in_dem)
        for layer in range(genetics_layers):
            Dem.append(max(in_dem, out_dem))
        Dem.append(out_dem)

        MatrixNet.__init__(self, Dem, [self.Mutability, -self.Mutability], activation, activation_der,
                           color_formula_param)

        if genetics_weights is not None:
            self.WeightArray = genetics_weights

    def compatable(self, evolvingNet2):
        return self.InDem == evolvingNet2.InDem and \
               self.OutDem == evolvingNet2.OutDem and \
               self.Layers == evolvingNet2.Layers

    def breed(self, evolvingNet2):
        assert self.compatable(evolvingNet2)
        newWeights = []
        for i in range(len(self.WeightArray)):
            newWeights.append([])
            for j in range(len(self.WeightArray[i])):
                newWeights[i].append([])
                for k in range(len(self.WeightArray[i][j])):
                    newWeights[i][j].append(random.choice([self.WeightArray[i][j][k],
                                                           evolvingNet2.WeightArray[i][j][k]]))
            newWeights[i] = numpy.array(newWeights[i])

        newNet = StaticEvolvingNet(self.InDem, self.OutDem, self.Layers, genetics_weights=newWeights)
        return newNet

    def replicate(self):
        newWeights = []
        for i in range(len(self.WeightArray)):
            newWeights.append([])
            for j in range(len(self.WeightArray[i])):
                newWeights[i].append([])
                for k in range(len(self.WeightArray[i][j])):
                    newWeights[i][j].append(self.mutate(self.WeightArray[i][j][k]))
            newWeights[i] = numpy.array(newWeights[i])

        newNet = StaticEvolvingNet(self.InDem, self.OutDem, self.Layers, genetics_weights=newWeights)
        return newNet

    def mutate(self, number):
        return number + (0.5 - random.random()) * 2.0 * self.Mutability

    def distance(self, net):
        return sum(map(distance_formula, self.WeightArray, net.WeightArray))
