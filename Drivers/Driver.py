import random
from abc import abstractmethod

import pygame
from numpy import mean, median
from typing import ClassVar, List, Type

from Nets import Net
from Nets.EvolvingNet import EvolvingNet
from Simulations import Simulation


class Driver:
    def __init__(self,
                 population_size: int,
                 simulation: Type[Simulation.Simulation],
                 generation_size: int,
                 row_size: int,
                 row_count: int,
                 mutability: float = 0.5,
                 evolving_class: Type[Net.Net] = EvolvingNet):
        self.in_dem: int = simulation.in_dem
        self.out_dem: int = simulation.in_dem
        self.row_size: int = row_size
        self.row_count: int = row_count
        self.population_size: int = population_size
        self.mutability: float = mutability
        self.simulation = simulation
        self.evolving_class = evolving_class
        self.generation_size: int = generation_size
        self.median: float = 0.0
        self.average: float = 0.0
        self.maximum: float = 0.0
        self.minimum: float = 0.0

        self.population: List[evolving_class] = []
        for i in range(self.population_size):
            self.population.append(self.evolving_class(self.in_dem, self.out_dem, 1, mutability=mutability))

    def run(self):
        self.simulation.restart()
        fitness = self.simulation.run_generations(self.population, self.generation_size)
        self.average = mean(fitness)
        self.median = median(fitness)
        self.maximum = max(fitness)
        self.minimum = min(fitness)
        self.repopulate(fitness)

    def test(self):
        fitness = self.simulation.run(self.population)
        self.average = mean(fitness)
        self.median = median(fitness)
        self.maximum = max(fitness)
        self.minimum = min(fitness)

    def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, dot_size: int = 10):
        for i in range(self.row_count * self.row_size):
            if (i < len(self.population)):
                self.population[i].update(screen,
                                          x + (i % self.row_size) * (width // self.row_size),
                                          y + (i // self.row_size) * (height // self.row_count),
                                          width // self.row_size,
                                          height // self.row_count,
                                          dot_size)
                self.population[i].draw()
                # print(self.population[i])

    def run_visual(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, dot_size: int = 10):
        self.simulation.restart()
        print("RUN " + ",".join(map(str, self.population)))
        fitness = self.simulation.run_generations_visual(self.population,
                                                         self.generation_size,
                                                         self.draw,
                                                         screen, x, y, width, height, dot_size)
        self.average = mean(fitness)
        self.median = median(fitness)
        self.maximum = max(fitness)
        self.minimum = min(fitness)

        self.repopulate(fitness)

    @abstractmethod
    def repopulate(self, fitness: List[float]):
        pass
