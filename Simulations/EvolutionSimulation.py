import os
import random
from typing import List, Type

import math
import numpy
import pygame

from Drivers.Driver import Driver
from Nets.Net import Net
from Simulations.Simulation import Simulation
from formulas import distance_formula


class EvolutionSimulation(Simulation):
    def __init__(self, imitator, in_dem, out_dem, layers):
        super(EvolutionSimulation, self).__init__(in_dem,
                                                  out_dem,
                                                  layers)
        self.imitator = imitator

        self.states = [[0, 1], [1, 1], [1, 0], [0, 0]]

    def restart(self):
        self.count = 0

    def run(self, population: List[Type[Net]]):

        fitness = []

        states_index = self.count % len(self.states)
        input_state = self.states[states_index]

        for net in population:
            net.set_in(input_state)
            fitness.append(
                numpy.linalg.norm(numpy.subtract(1.0, abs(numpy.subtract(net.get_out(), self.imitator(input_state))))))

        self.count += 1

        return numpy.array(fitness)

    def run_generations(self,
                        population: List[Type[Net]],
                        generation: int) -> numpy.array:
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
                               dot_size: int = 10) -> numpy.array:
        fitness = numpy.zeros((len(population)))
        for i in range(generation):
            fitness = numpy.add(fitness, self.run(population))

            driver.draw(screen, x, y, width, height, dot_size=dot_size)
            pygame.display.flip()

        fitness = numpy.divide(fitness, generation)
        return fitness
