import numpy, random, math, pygame

def sigmoid(x):
    return 1.0 / (1.0 + math.e ** -float(x))



sigmoid_Array = numpy.vectorize(sigmoid)


class MatrixNet:
    def __init__(self, Dem, weight_range):
        self.InputArray = numpy.array([[0]] * Dem[0])
        self.WeightArray = []
        self.NodesValueArray = []
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
                    self.InputArray[i][0] = array[i]

        # print self.InputArray

    def getOut(self):
        ## self.NodesValueArray[0] = sigmoid_Array(self.WeightArray[0].dot(numpy.reshape(numpy.append(self.InputArray, -1),((len(self.InputArray) + 1), 1) )))
        print(self.WeightArray)
        self.NodesValueArray[0] = sigmoid_Array(self.WeightArray[0].dot(self.InputArray))
        # self.NodesValueArray[0][-1] = -1
        for i in range(1, len(self.NodesValueArray)):
            self.NodesValueArray[i] = sigmoid_Array(self.WeightArray[i].dot(self.NodesValueArray[i -1]))
            # self.NodesValueArray[i][-1] = -1
        return self.NodesValueArray[-1]

    def getOutThreshold(self):
        # print self.WeightArray
        self.NodesValueArray[0] = sigmoid_Array(self.WeightArray[0].dot(numpy.reshape(numpy.append(self.InputArray, -1),((len(self.InputArray) + 1), 1))))

        # self.NodesValueArray[0] = sigmoid_Array(self.WeightArray[0].dot(self.InputArray))
        # self.NodesValueArray[0][-1] = -1
        for i in range(1, len(self.NodesValueArray)):
            # self.NodesValueArray[i] = sigmoid_Array(self.WeightArray[i].dot(self.NodesValueArray[i -1]))
            self.NodesValueArray[i] = sigmoid_Array(self.WeightArray[i].dot(
                numpy.reshape(numpy.append(self.NodesValueArray[i -1], -1), ((len(self.NodesValueArray[i - 1]) + 1), 1))))

            # self.NodesValueArray[i][-1] = -1
        return self.NodesValueArray[-1]

    def learnWithThreshold(self, ratio, target):
        l = len(target)

        target = numpy.reshape(numpy.array([target]), (l, 1))
        print("Target", target)
        print("Value", self.NodesValueArray[-1])
        past = target - self.NodesValueArray[-1]
        print(past)
        # print target
        for i in range(len(self.NodesValueArray) - 1, 0, -1):
            # for past_row in past:
            NodesValueArraytemp = self.NodesValueArray[i]#numpy.reshape(numpy.append(self.NodesValueArray[i], -1), ((len(self.NodesValueArray[i]) + 1), 1))
            NodesValueArraytemp2 = numpy.reshape(numpy.append(self.NodesValueArray[i - 1], -1), (1, len(self.NodesValueArray[i - 1]) + 1))

            current = ((numpy.array([[1.0]] * len(NodesValueArraytemp)) - NodesValueArraytemp) *
                       NodesValueArraytemp).dot(NodesValueArraytemp2)

            current = current * past
            past = (numpy.array([[1] * len(current)])).dot(current)
            # print "-----"

            # print past
            # past /= 4
            # print past

            past = numpy.reshape(past, (len(past[0]), 1))[:-1]

            current = current * ratio
            self.WeightArray[i] = self.WeightArray[i] + current
            print("Weight array", i, self.WeightArray[i])
        NodesValueArraytemp = self.NodesValueArray[0]
        NodesValueArraytemp2 = numpy.reshape(numpy.append(self.InputArray, -1),
                                             (1, len(self.InputArray) + 1))

        current = ((numpy.array([[1.0]] * len(NodesValueArraytemp)) - NodesValueArraytemp) *
                   NodesValueArraytemp).dot(NodesValueArraytemp2)

        current = current * ratio
        self.WeightArray[0] = self.WeightArray[0] + current
        print("Weight array", 0, self.WeightArray[0])
    def learn(self, ratio, target):
        l = len(target)

        target = numpy.reshape(numpy.array([target]), (l, 1))
        past = target - self.NodesValueArray[-1]

        for i in range(len(self.NodesValueArray) - 1, 0, -1):
            # for past_row in past:
            current = ((numpy.array([[1.0]] * len(self.NodesValueArray[i])) - self.NodesValueArray[i]) *
                       self.NodesValueArray[i]).dot(numpy.reshape(self.NodesValueArray[i- 1], (1, len(self.NodesValueArray[i- 1]))))

            current = current * past
            past = (numpy.array([[1] * len(current)])).dot(current)

            past = numpy.reshape(past,(len(past[0]),1))

            current = current * ratio
            self.WeightArray[i] = self.WeightArray[i] + current


        current = ((numpy.array([[1.0]] * len(self.NodesValueArray[0])) - self.NodesValueArray[0]) *
                   self.NodesValueArray[0]).dot(
            numpy.reshape(self.InputArray, (1, len(self.InputArray)))) * past

        current = current * ratio
        self.WeightArray[0] = self.WeightArray[0] + current
    def draw(self, screen, x, y, scale):
        for y_ in range(0, len(self.InputArray)):
            pygame.draw.circle(screen, [int(self.InputArray[y_] * 255.)] * 3, [x, y + y_ * scale],
                               scale / 10)
        for x_ in range(0, len(self.NodesValueArray)):
            for y_ in range(0, len(self.NodesValueArray[x_])):
                pygame.draw.circle(screen, [int(self.NodesValueArray[x_][y_] * 255.)] * 3, [x + (x_ + 1) * scale, y + y_ * scale], scale / 10)
                for y2 in range(0, len(self.WeightArray[x_][y_])):
                    pygame.draw.line(screen, [255. - 255. * sigmoid(self.WeightArray[x_][y_][y2]),
                                              255. * sigmoid(self.WeightArray[x_][y_][y2]), 0],
                                     [x + (x_ + 1) * scale, y + y_ * scale], [x + (x_) * scale, y + y2 * scale])

