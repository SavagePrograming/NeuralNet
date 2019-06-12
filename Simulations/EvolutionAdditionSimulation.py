from typing import Type, List

from Drivers.Driver import Driver
from Nets.Net import Net
import numpy
import pygame

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
    for i in range(0, len(bits)):
        num += (2 ** i) * bits[i]
    return num


def add(arr):
    return numtobits((bitstonum(arr[:3]) + bitstonum(arr[3:])), 4)


class AdditionSimulation(Simulation):
    def __init__(self):
        super(AdditionSimulation, self).__init__(6, 4, 2)

        self.imitator = add
        self.count = 0

        self.states = []
        for i in range(0, 2 ** 6):
            self.states.append(numtobits(i, 6))

    def restart(self):
        pass

    def run(self, population: List[Type[Net]]):

        fitness = []

        states_index = self.count % len(self.states)
        input_state = self.states[states_index]

        for net in population:
            net.set_in(input_state)
            fitness.append(numpy.linalg.norm(numpy.subtract(1.0,
                                                            numpy.abs(numpy.subtract(net.get_out(),
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

            driver.draw(screen, x, y, width, height, dot_size=dot_size)
            pygame.display.flip()

        fitness = numpy.divide(fitness, generation)
        return fitness
