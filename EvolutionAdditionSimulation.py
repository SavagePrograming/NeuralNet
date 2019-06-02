import os
import random

import math
import numpy
import pygame


def numtobits(num, bits):
    l = [0] * bits
    num = num % (2 ** bits)
    for i in range(bits - 1, -1, -1):
        l[i] = int(num / (2 ** i))
        num = num % (2 ** i)
    return l


def bitstonum(bits):
    num = 0
    for i in range(0, len(bits)):
        num += (2 ** i) * bits[i]
    return num

def add(arr):
    return numtobits((bitstonum(arr[:3]) + bitstonum(arr[3:])), 4)

from formulas import distance_formula

class AdditionSimulation:
    def __init__(self):
        self.InDem = 6
        self.OutDem = 4


        self.Imitator = add
        self.count = 0

        self.States = []
        for i in range(0,2 ** 6):
            self.States.append(numtobits(i, 6))

    def restart(self):
        pass


    def run(self, population):

        Fitness = []

        StatesIndex = (self.count) % len(self.States)
        Input = self.States[StatesIndex]

        for Net in population:
            Net.setIn(Input)
            Fitness.append(numpy.linalg.norm(numpy.subtract(1.0, numpy.abs(numpy.subtract(Net.get_out(), self.Imitator(Input))))))

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

            # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:
            #         pass
        Fitness /= generation
        return Fitness