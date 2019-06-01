import os

import numpy
import pygame


class EvolveSimulation:
    def __init__(self, imitator, in_dem, out_dem):
        self.InDem = in_dem
        self.OutDem = out_dem
        self.Imitator = imitator
        self.count = 0

        self.States = [[0,1], [1,1],[1,0],[0,0]]

    def restart(self):
        self.count = 0


    def run(self, population):

        Fitness = []

        StatesIndex = (self.count) % len(self.States)
        Input = self.States[StatesIndex]

        for Net in population:
            Net.setIn(Input)
            Fitness.append(numpy.linalg.norm(1.0 - abs(Net.getOutThreshold() - self.Imitator(Input))))

        self.count += 1

        return numpy.array(Fitness)

    def run_generations(self, population, generation):
        Fitness = numpy.zeros((len(population)))
        for i in range(generation):
            Fitness += self.run(population)
        Fitness /= generation
        return Fitness

    def run_generations_visual(self, population, generation, driver, screen, row_size, row_count, x, y, width, height, dot_size=10):
        Fitness = numpy.zeros((len(population)))
        for i in range(generation):

            Fitness += self.run(population)

            driver.draw(screen, row_size, row_count, x, y, width, height, dot_size=10)
            pygame.display.flip()

        Fitness /= generation
        return Fitness