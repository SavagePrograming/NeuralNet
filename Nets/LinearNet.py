from functools import total_ordering

import numpy, random, math, pygame


from formulas import distance_formula, sigmoid, sigmoid_der


def color_formula(x):
    return [0, int(x * 255.), 0]


class LinearNet:
    def __init__(self, in_dem, out_dem, middle_dem, weights, activation=sigmoid, activation_der=sigmoid_der,
                 color_formula_param=color_formula):

        self.in_dem = in_dem
        self.out_dem = out_dem
        self.middle_dem = middle_dem

        self.Score = 0
        self.input_nodes = numpy.zeros((1, in_dem))

        self.node_values = numpy.zeros((1, middle_dem + out_dem))
        self.node_sum = numpy.zeros((1, middle_dem + out_dem))

        self.weights = weights

        self.activation_function = activation
        self.activation_derivative = activation_der

        self.color_formula = color_formula_param

    def set_in(self, array):
        assert len(array) == self.in_dem
        self.input_nodes = numpy.array(array, ndmin=2)

    def get_out(self):
        self.node_sum = numpy.dot(self.input_nodes, self.weights[:self.in_dem])
        for i in range(self.middle_dem):
            self.node_values[0][i] = sigmoid(self.node_sum[0][i])
            self.node_sum = numpy.add(self.node_sum, numpy.multiply(self.node_values[0][i], self.weights[i:i+1]))
        self.node_values = self.activation_function(self.node_sum)
        return self.node_values[0][self.middle_dem:]

    def learn(self, ratio, target):
        pass
    def draw(self, screen, x, y, width, height, scale_dot=5):
        in_spacing = (height - scale_dot * 2) // self.in_dem
        for i in range(self.in_dem):
            pygame.draw.circle(screen,
                               self.ColorFormula(self.input_nodes[0][i]),
                               [int(x + scale_dot),
                                int(y + scale_dot + i * in_spacing)],
                               int(scale_dot))

        out_spacing = (height - scale_dot * 2) // self.out_dem
        for i in range(self.out_dem):
            pygame.draw.circle(screen,
                           self.ColorFormula(self.out_dem[0][i]),
                           [int(x + width - scale_dot),
                            int(y + scale_dot + i * out_spacing)],
                           int(scale_dot))

    def __eq__(self, other):
        return self.Score == other.Score

    def __lt__(self, other):
        return self.Score < other.Score

    def __gt__(self, other):
        return self.Score > other.Score

    def __ge__(self, other):
        return self.Score >= other.Score

    def __le__(self, other):
        return self.Score <= other.Score

    def __add__(self, other):
        if isinstance(other, LinearNet):
            return self.Score + other.Score
        else:
            return self.Score + other

    def __radd__(self, other):
        if isinstance(other, LinearNet):
            return self.Score + other.Score
        else:
            return self.Score + other