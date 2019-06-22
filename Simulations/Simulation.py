from abc import abstractmethod
from typing import List, Type, Callable

from Nets.Net import Net
import numpy
import pygame


class Simulation:
    def __init__(self,
                 in_dem: int,
                 out_dem: int,
                 layers: int):
        self.in_dem = in_dem
        self.out_dem = out_dem
        self.layers = layers
        self.count = 0

    def restart(self):
        self.count = 0

    @abstractmethod
    def run(self, population: List[Type[Net]]) -> numpy.array:
        pass

    @abstractmethod
    def run_generations(self,
                        population: List[Type[Net]],
                        generation: int) -> numpy.array:
        pass

    @abstractmethod
    def run_generations_visual(self,
                               population: List[Type[Net]],
                               generation: int,
                               driver_draw: Callable,
                               screen: pygame.Surface,
                               x: int,
                               y: int,
                               width: int,
                               height: int,
                               dot_size: int = 10) -> numpy.array:
        pass
