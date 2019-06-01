import random

from math import ceil
from numpy import mean, median

from EvolvingNet import EvolvingNet
def clean(a):
    i = 0
    while i < len(a):
        if not a[i]:
            del a[i]
        else:
            i+= 1

class EvolutionSpeciationDriver:
    def __init__(self, population_size, survivor_ratio, simulation,
                 generation_size, species_threshold, species_independent_survivor_ratio, mutability=0.5,
                 evolving_class=EvolvingNet):
        self.SpeciesThreshold = species_threshold
        self.InDem = simulation.InDem
        self.OutDem = simulation.OutDem
        self.PopulationSize = population_size
        self.Mutability = mutability
        self.SurvivorRatio = survivor_ratio
        self.Simulation = simulation
        self.EvolvingClass = evolving_class
        self.GenerationSize = generation_size
        self.Median = 0.0
        self.Average = 0.0
        self.Maximum = 0.0
        self.Minimum = 0.0
        self.Species = []

        self.Population = []
        for i in range(self.PopulationSize):
            child = self.EvolvingClass(self.InDem, self.OutDem, 1, mutability=mutability)
            self.Population.append(child)
            self.add_to_specie(child)

    def run(self):
        self.Simulation.restart()
        Fitness = self.Simulation.run_generations(self.Population, self.GenerationSize)
        self.Average = mean(Fitness)
        self.Median = median(Fitness)
        self.Maximum = max(Fitness)
        self.Minimum = min(Fitness)
        self.repopulate(Fitness)

    def test(self):
        Fitness = self.Simulation.run(self.Population)
        self.Average = mean(Fitness)
        self.Median = median(Fitness)
        self.Maximum = max(Fitness)
        self.Minimum = min(Fitness)

    def draw(self, screen, row_size, row_count, x, y, width, height, dot_size=10):
        for i in range(row_count * row_size):
            self.Population[i].draw(screen,
                                    x + (i % row_size) * (width // row_size),
                                    y + (i // row_size) * (height // row_count),
                                    width // row_size,
                                    height // row_count,
                                    dot_size)

    def run_visual(self, screen, row_size, row_count, x, y, width, height, dot_size=10):
        self.Simulation.restart()
        Fitness = self.Simulation.run_generations_visual(self.Population, self.GenerationSize, self, screen, row_size,
                                                       row_count, x, y, width, height, dot_size)
        self.Average = mean(Fitness)
        self.Median = median(Fitness)
        self.Maximum = max(Fitness)
        self.Minimum = min(Fitness)
        self.repopulate(Fitness)

    def draw(self, screen, row_size, row_count, x, y, width, height, dot_size=10):
        for i in range(row_count * row_size):
            self.Population[i].draw(screen,
                                    x + (i % row_size) * (width // row_size),
                                    y + (i // row_size) * (height // row_count),
                                    width // row_size,
                                    height // row_count,
                                    dot_size)

    def repopulate(self, Fitness):
        survivors = int(self.PopulationSize * self.SurvivorRatio)
        for i in range(len(Fitness)):
            self.Population[i].Score = Fitness[i]
        MAX = max(self.Population)
        self.Population.remove(MAX)
        for s in range(len(self.Species)):
            size = len(self.Species[s])
            for net in self.Species[s]:
                net.Score /= ceil(size / 4)
        self.Population.sort(reverse=True)
        self.Population = self.Population[:survivors - 1]
        self.Population.insert(0, MAX)
        for s in range(len(self.Species)):
            n = 0
            while n < len(self.Species[s]):
                if self.Species[s][n] not in self.Population:
                    del self.Species[s][n]
                else:
                    n += 1
        clean(self.Species)
        self.speciate(self.PopulationSize - survivors)

    def speciate(self, size):
        # print(self.Species)
        survivors = len(self.Population)
        for i in range(size):
            child = self.Population[i % survivors].breed(random.choice(self.Population[:survivors]))
            self.Population.append(child)
            self.add_to_specie(child)
        if not len(self.Species) > self.PopulationSize:
            print("Species:" + str(len(self.Species)))

    def add_to_specie(self, child):
        remaining = True
        for s in range(len(self.Species)):
            # print(child.distance(self.Species[s][0]))
            if child.distance(self.Species[s][0]) < self.SpeciesThreshold:
                self.Species[s].append(child)
                remaining = False
                break
        if remaining:
            self.Species.append([child])