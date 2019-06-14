from typing import Callable, List, Type

import numpy
import pygame

from Drivers.Driver import Driver
from Nets.Net import Net
from Simulations.EvolutionSimulation import EvolutionSimulation
from Simulations.Simulation import Simulation


def numtobits(num, bits):
    l = [0] * bits
    num = num % (2 ** bits)
    for i in range(bits - 1, -1, -1):
        l[i] = int(num / (2 ** i))
        num = num % (2 ** i)
    return l


def bitstonum(bits):
    num = 0
    for i in range(len(bits)):
        num += (2 ** i) * bits[i]
    return num


def add(arr):
    return numpy.ceil(numpy.divide(numpy.add(arr[:3], arr[3:]), 2.0))


class DivisionSimulation(Simulation):
    def __init__(self, layers: int, row_count: int, row_size: int):
        super(DivisionSimulation, self).__init__(6, 3, layers)
        self.imitator: Callable = add
        self.row_count: int = row_count
        self.row_size: int = row_size
        self.states: List[List[int]] = []
        for i in range(2 ** 6):
            self.states.append(numtobits(i, 6))

    def restart(self):
        pass

    def run(self, population: List[Type[Net]]):

        fitness = []

        states_index = self.count % len(self.states)
        input_state = self.states[states_index]

        for Net in population:
            Net.set_in(input_state)
            fitness.append(numpy.linalg.norm(numpy.subtract(1.0,
                                                            numpy.abs(numpy.subtract(Net.get_out(),
                                                                                     self.imitator(input_state))))))

        self.count += 1

        return numpy.array(fitness)

    def run_generations(self,
                        population: List[Type[Net]],
                        generation: int):
        fitness = numpy.zeros((len(population)))
        for i in range(generation):
            fitness = numpy.add(fitness, self.run(population))
        fitness = numpy.divide(fitness, generation)
        return fitness

    def run_generations_visual(self,
                               population: List[Type[Net]],
                               generation: int,
                               driver: Type[Driver],
                               screen: pygame.Surface,
                               x: int,
                               y: int,
                               width: int,
                               height: int,
                               dot_size: int = 10):
        fitness = numpy.zeros((len(population)))
        for i in range(generation):
            fitness = numpy.add(fitness, self.run(population))

            driver.draw(screen, self.row_size, self.row_count, x, y, width, height, dot_size=dot_size)

            pygame.display.flip()

        fitness = numpy.divide(fitness, generation)
        return fitness
