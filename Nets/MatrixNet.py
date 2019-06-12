from functools import total_ordering

import numpy, random, math, pygame


from formulas import distance_formula, sigmoid, sigmoid_der


def color_formula(x):
    return [0, int(x * 255.), 0]


class MatrixNet:
    def __init__(self, Dem, weight_range, activation=sigmoid, activation_der=sigmoid_der,
                 color_formula_param=color_formula):
        self.InputArray = numpy.array([[0]] * Dem[0])
        self.WeightArray = []
        self.NodesValueArray = []
        self.ActivationFunction = activation
        self.activation_function_derivative = activation_der
        self.ColorFormula = color_formula_param
        self.Dimensions = Dem
        self.Score = 0

        for i in range(1, len(Dem)):
            Node_array = []
            Weight_array = []

            for ii in range(0, Dem[i]):
                Node_array.append([0])
                Weight_array.append([])
                for iii in range(0, Dem[i - 1] + 1):
                    Weight_array[ii].append(random.random() * (weight_range[1] - weight_range[0]) + weight_range[0])

            self.WeightArray.append(numpy.array(Weight_array))
            self.NodesValueArray.append(numpy.array(Node_array))

    def __init__Origional(self, Dem, weight_range):
        self.InputArray = numpy.array([[0]] * Dem[0])
        self.WeightArray = []
        self.NodesValueArray = []
        for i in range(1, len(Dem)):
            Node_array = []
            Weight_array = []

            for ii in range(0, Dem[i]):
                Node_array.append([0])
                Weight_array.append([])
                for iii in range(0, Dem[i - 1]):
                    Weight_array[ii].append(random.random() * (weight_range[1] - weight_range[0]) + weight_range[0])

            self.WeightArray.append(numpy.array(Weight_array))
            self.NodesValueArray.append(numpy.array(Node_array))

    def set_in(self, array):
        if len(array) == len(self.InputArray):
            for i in range(0, len(array)):
                if array[i] is not None:
                    self.InputArray[i][0] = array[i]

    def get_out(self):
        self.NodesValueArray[0] = self.ActivationFunction(
            self.WeightArray[0].dot(numpy.reshape(numpy.append(self.InputArray, 1.0), ((len(self.InputArray) + 1), 1))))

        for i in range(1, len(self.NodesValueArray)):
            self.NodesValueArray[i] = self.ActivationFunction(self.WeightArray[i].dot(
                numpy.reshape(numpy.append(self.NodesValueArray[i - 1], 1.0),
                              ((len(self.NodesValueArray[i - 1]) + 1), 1))))
        return self.NodesValueArray[-1]

    def learn(self, ratio, target):
        l = len(target)

        target = numpy.reshape(numpy.array([target]), (l, 1))

        past = numpy.multiply(2.0, (numpy.subtract(target, self.NodesValueArray[-1])))

        error = distance_formula(target, self.NodesValueArray[-1])

        for i in range(len(self.NodesValueArray) - 1, 0, -1):

            NodesValueArraytemp = self.NodesValueArray[i]

            NodesValueArraytemp2 = numpy.reshape(numpy.append(self.NodesValueArray[i - 1], 1),
                                                 (1, len(self.NodesValueArray[i - 1]) + 1))

            sigder = self.activation_function_derivative(NodesValueArraytemp)
            sigder_with_past = numpy.multiply(sigder, past)
            current = sigder_with_past.dot(NodesValueArraytemp2)
            past = numpy.transpose(sigder_with_past).dot(self.WeightArray[i])
            past = numpy.reshape(past, (len(past[0]), 1))[:-1]
            current = numpy.multiply(current, ratio)
            self.WeightArray[i] = numpy.add(self.WeightArray[i], current)

        NodesValueArraytemp = self.NodesValueArray[0]

        NodesValueArraytemp2 = numpy.reshape(numpy.append(self.InputArray, 1),
                                             (1, len(self.InputArray) + 1))
        sig = self.activation_function_derivative(NodesValueArraytemp)
        sig_with_past = numpy.multiply(sig, past)

        current = sig_with_past.dot(NodesValueArraytemp2)
        current = numpy.multiply(current, ratio)
        self.WeightArray[0] = numpy.add(self.WeightArray[0], current)

        return error

    def draw(self, screen, x, y, width, height, scale_dot=5):
        scale_y = (height - scale_dot * 2) // max(self.Dimensions)
        scale_x = (width - scale_dot * 2) // (len(self.Dimensions) - 1)
        for y_ in range(0, len(self.InputArray)):
            pygame.draw.circle(screen, self.ColorFormula(self.InputArray[y_]),
                               [int(x + scale_dot), int(y + scale_dot + y_ * scale_y)],
                               int(scale_dot))
        for x_ in range(0, len(self.NodesValueArray)):
            for y_ in range(0, len(self.NodesValueArray[x_])):
                pygame.draw.circle(screen, self.ColorFormula(self.NodesValueArray[x_][y_]),
                                   [int(x + scale_dot + (x_ + 1) * scale_x), int(y + scale_dot + y_ * scale_y)],
                                   int(scale_dot))
                for y2 in range(0, len(self.WeightArray[x_][y_])):
                    pygame.draw.line(screen,
                                     [255. - 255. * self.ActivationFunction(self.WeightArray[x_][y_][y2]), 125
                                      , 255. * self.ActivationFunction(self.WeightArray[x_][y_][y2])],
                                     [x + scale_dot + (x_ + 1) * scale_x, y + scale_dot + y_ * scale_y],
                                     [x + scale_dot + (x_) * scale_x, y + scale_dot + y2 * scale_y])

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
        if isinstance(other, MatrixNet):
            return self.Score + other.Score
        else:
            return self.Score + other

    def __radd__(self, other):
        if isinstance(other, MatrixNet):
            return self.Score + other.Score
        else:
            return self.Score + other