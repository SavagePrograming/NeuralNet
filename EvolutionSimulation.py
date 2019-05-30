import numpy


class EvolveSimulation:
    def __init__(self, imitator):
        self.InDem = 4
        self.OutDem = 4
        self.Imitator = imitator
        self.States = [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                          [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                          [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],

                          [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                          [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                          [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                          [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],

                          [1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                          [0.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                          [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 1.0, 0.0, 1.0],
                          [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
                          [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],

                          [0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
                          [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                          [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
                          [1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
                          [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                          [0.0, 0.0, 1.0, 1.0, 0.0, 1.0],

                          [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                          [0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
                          [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
                          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                          [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                          [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]]

    def restart(self):
        self.count = 0


    def run(self, population):

        Error = []

        StatesIndex = (self.count) % len(self.States)
        Input = self.States[StatesIndex]

        for Net in population:
            Net.setIn(Input)
            Error.append(numpy.linalg.norm(Net.getOutThreshold() - self.Imitator(Input)))

        self.count += 1

        return numpy.array(Error)

    def run_generations(self, population, generation):
        Error = numpy.zeros((len(population)))
        for i in range(generation):
            Error += self.run(population)
        Error /= generation
        return Error

