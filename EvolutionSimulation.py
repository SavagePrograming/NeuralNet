import os

import numpy
import pygame


class EvolveSimulation:
    def __init__(self, imitator):
        self.InDem = 2
        self.OutDem = 2
        self.Imitator = imitator
        self.count = 0

        self.States = [[0,1], [1,1],[1,0],[0,0]]

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

    def run_generations_visual(self, population, generation, driver, screen, row_size, row_count, x, y, width, height, dot_size=10):
        Error = numpy.zeros((len(population)))
        for i in range(generation):
            # print(i)
            Error += self.run(population)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                   os.quit()
            # screen.fill([0, 0, 100])
            driver.draw(screen, row_size, row_count, x, y, width, height, dot_size=10)
            pygame.display.flip()

        Error /= generation
        return Error