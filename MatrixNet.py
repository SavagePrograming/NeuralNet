from functools import total_ordering

import numpy, random, math, pygame


def sigmoid(x):
    return 1.0 / (1.0 + math.e ** -float(x))


sigmoid_Array = numpy.vectorize(sigmoid)


def sigmoid_der(x):
    return (1.0 - x) * x


sigmoid_der_Array = numpy.vectorize(sigmoid_der)


def color_formula(x):
    return [255 - int(x * 255.), 255, 255 - int(x * 255.)]


class MatrixNet:
    def __init__(self, Dem, weight_range, activation=sigmoid_Array, activation_der=sigmoid_der_Array,
                 color_formula=color_formula):
        self.InputArray = numpy.array([[0]] * Dem[0])
        self.WeightArray = []
        self.NodesValueArray = []
        self.ActivationFunction = activation
        self.ActivationFunctionDerivitive = activation_der
        self.ColorFormula = color_formula
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

    def setIn(self, array):
        if len(array) == len(self.InputArray):
            for i in range(0, len(array)):
                if array[i] is not None:
                    self.InputArray[i][0] = float(array[i])

        # print self.InputArray

    def getOut(self):
        ## self.NodesValueArray[0] = sigmoid_Array(self.WeightArray[0].dot(numpy.reshape(numpy.append(self.InputArray, -1),((len(self.InputArray) + 1), 1) )))
        # print(self.WeightArray)
        self.NodesValueArray[0] = self.ActivationFunction(self.WeightArray[0].dot(self.InputArray))
        # self.NodesValueArray[0][-1] = -1
        for i in range(1, len(self.NodesValueArray)):
            self.NodesValueArray[i] = self.ActivationFunction(self.WeightArray[i].dot(self.NodesValueArray[i - 1]))
            # self.NodesValueArray[i][-1] = -1
        return self.NodesValueArray[-1]

    def getOutThreshold(self):
        # print self.WeightArray
        # print("++++++")
        self.NodesValueArray[0] = self.ActivationFunction(
            self.WeightArray[0].dot(numpy.reshape(numpy.append(self.InputArray, 1), ((len(self.InputArray) + 1), 1))))
        # print("Values[0]:" + str(self.NodesValueArray[0]))
        # print("Values[0]:" + str(self.NodesValueArray[0]))
        # print("Values[0]:"+str(self.NodesValueArray[0]))
        # self.NodesValueArray[0] = sigmoid_Array(self.WeightArray[0].dot(self.InputArray))
        # self.NodesValueArray[0][-1] = -1
        for i in range(1, len(self.NodesValueArray)):
            # self.NodesValueArray[i] = sigmoid_Array(self.WeightArray[i].dot(self.NodesValueArray[i -1]))
            self.NodesValueArray[i] = self.ActivationFunction(self.WeightArray[i].dot(
                numpy.reshape(numpy.append(self.NodesValueArray[i - 1], 1),
                              ((len(self.NodesValueArray[i - 1]) + 1), 1))))
            # print("Values:"+str(self.NodesValueArray[i]))
            # self.NodesValueArray[i][-1] = -1
        return self.NodesValueArray[-1]

    def learnWithThreshold(self, ratio, target):

        # print(self.InputArray)
        # print(target)
        # print("--------------")

        l = len(target)

        target = numpy.reshape(numpy.array([target]), (l, 1))
        # print("Target:" +str(target))
        # print("Real:"+str(self.NodesValueArray[-1]))
        past = 2 * (target - self.NodesValueArray[-1])
        error = numpy.linalg.norm(past)
        # print("Inital dif:" + str(past))
        # print target
        for i in range(len(self.NodesValueArray) - 1, 0, -1):
            # for past_row in past:
            NodesValueArraytemp = self.NodesValueArray[
                i]  # numpy.reshape(numpy.append(self.NodesValueArray[i], -1), ((len(self.NodesValueArray[i]) + 1), 1))
            # print("NodesValueArraytemp:" + str(NodesValueArraytemp))

            NodesValueArraytemp2 = numpy.reshape(numpy.append(self.NodesValueArray[i - 1], 1),
                                                 (1, len(self.NodesValueArray[i - 1]) + 1))
            # print("NodesValueArraytemp2:" + str(NodesValueArraytemp2))

            sigder = self.ActivationFunctionDerivitive(NodesValueArraytemp)
            # print("sigmoid div:" + str(sigder))

            sigder_with_past = sigder * past
            # print("sigder_with_past:" + str(sigder))

            current = sigder_with_past.dot(NodesValueArraytemp2)
            # print("current:" + str(current))

            # print("weights:" + str(self.WeightArray[i]))
            past = numpy.transpose(sigder_with_past).dot(self.WeightArray[i])
            # past = (numpy.array([[1] * len(current)])).dot(current)
            # print("|-----")

            # print("past:" + str(past))
            # print("other past:" + str(other_past))
            # past = other_past
            # past /= 4
            # print past

            past = numpy.reshape(past, (len(past[0]), 1))[:-1]
            # print("past reshaped:" + str(past))

            current = current * ratio
            # print("Weight shifts dif:" + str(current))

            self.WeightArray[i] = self.WeightArray[i] + current
            # print("Weight array", i, self.WeightArray[i])
        NodesValueArraytemp = self.NodesValueArray[0]
        # print("Current Value:" + str(NodesValueArraytemp))
        NodesValueArraytemp2 = numpy.reshape(numpy.append(self.InputArray, 1),
                                             (1, len(self.InputArray) + 1))
        # print("Current Value with thresh:" + str(NodesValueArraytemp2))

        sig = self.ActivationFunctionDerivitive(NodesValueArraytemp)

        # print("new past?:" + str(self.ActivationFunctionDerivitive(NodesValueArraytemp).dot(self.WeightArray[0])))
        sig_with_past = sig * past

        current = sig_with_past.dot(NodesValueArraytemp2)

        # print("Weight shifts with past:" + str(current))

        current = current * ratio
        # print("Weight shifts post ration:" + str(current))
        self.WeightArray[0] = self.WeightArray[0] + current
        # print("Weight array", 0, self.WeightArray[0])

        return error

    def learn(self, ratio, target):
        l = len(target)

        target = numpy.reshape(numpy.array([target]), (l, 1))
        past = target - self.NodesValueArray[-1]

        for i in range(len(self.NodesValueArray) - 1, 0, -1):
            # for past_row in past:
            sig = ((numpy.array([[1.0]] * len(self.NodesValueArray[i])) - self.NodesValueArray[i]) *
                   self.NodesValueArray[i])

            current = self.ActivationFunctionDerivitive(self.NodesValueArray[i]).dot(
                numpy.reshape(self.NodesValueArray[i - 1], (1, len(self.NodesValueArray[i - 1]))))

            current = current * past
            past = (numpy.array([[1] * len(current)])).dot(current)

            past = numpy.reshape(past, (len(past[0]), 1))

            current = current * ratio
            self.WeightArray[i] = self.WeightArray[i] + current

        current = self.ActivationFunctionDerivitive(self.NodesValueArray[0]).dot(
            numpy.reshape(self.InputArray, (1, len(self.InputArray)))) * past

        current = current * ratio
        self.WeightArray[0] = self.WeightArray[0] + current

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