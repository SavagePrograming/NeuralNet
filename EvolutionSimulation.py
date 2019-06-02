import os
import random

import math
import numpy
import pygame

from formulas import distance_formula

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
            Fitness.append(numpy.linalg.norm(numpy.subtract(1.0, abs(numpy.subtract(Net.get_out(), self.Imitator(Input))))))

        self.count += 1

        return numpy.array(Fitness)

    def run_generations(self, population, generation):
        fitness = numpy.zeros((len(population)))
        for i in range(generation):
            fitness = numpy.add(fitness, self.run(population))
        fitness = numpy.divide(fitness, generation)
        return fitness

    def run_generations_visual(self, population, generation, driver, screen, row_size, row_count, x, y, width, height, dot_size=10):
        fitness = numpy.zeros((len(population)))
        for i in range(generation):

            fitness = numpy.add(fitness, self.run(population))

            driver.draw(screen, row_size, row_count, x, y, width, height, dot_size=dot_size)
            pygame.display.flip()

            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:
            #         pass
        fitness = numpy.divide(fitness, generation)
        return fitness
