import random

from math import ceil
from numpy import mean, median, math

from Nets.EvolvingNet import EvolvingNet


def clean(a):
    i = 0
    while i < len(a):
        if not a[i]:
            del a[i]
        else:
            i += 1


class EvolutionSpeciationDriver:
    def __init__(self, population_size, survivor_ratio, simulation,
                 generation_size, species_threshold, species_independent_survivor_ratio, balancing_focus,
                 mutability=0.5,
                 evolving_class=EvolvingNet):
        self.BalanceTop = .5
        self.BalanceBottom = .25
        self.SpeciesThreshold = species_threshold
        self.BalanceFocus = balancing_focus
        self.SISR = species_independent_survivor_ratio
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
            child = self.EvolvingClass(self.InDem, self.OutDem, self.Simulation.Layers, mutability=mutability)
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

        row_count = int(math.sqrt(len(self.Species)))
        row_size = math.ceil(len(self.Species) / row_count)
        for i in range(row_count * row_size):
            if len(self.Species) > i:
                self.Species[i][0].draw(screen,
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

    def repopulate(self, Fitness):
        survivors = int(self.PopulationSize * self.SurvivorRatio) - ceil(self.PopulationSize * self.SISR)
        for i in range(len(Fitness)):
            self.Population[i].Score = Fitness[i]
        SIS = []
        if self.SISR < .25:
            for i in range(ceil(self.PopulationSize * self.SISR)):
                MAX = max(self.Population)
                self.Population.remove(MAX)
                SIS.append(MAX)
        else:
            self.Population.sort(reverse=True)
            SIS = self.Population[:ceil(self.PopulationSize * self.SISR)]
            self.Population = self.Population[ceil(self.PopulationSize * self.SISR):]

        for s in range(len(self.Species)):
            size = len(self.Species[s])
            for net in self.Species[s]:
                net.Score /= ceil(size / 4)
        self.Population.sort(reverse=True)
        self.Population = self.Population[:survivors]
        self.Population = SIS + self.Population
        for s in range(len(self.Species)):
            n = 0
            while n < len(self.Species[s]):
                if self.Species[s][n] not in self.Population:
                    del self.Species[s][n]
                else:
                    n += 1
        if self.BalanceFocus == 0:
            clean(self.Species)
            self.speciate(self.PopulationSize - int(self.PopulationSize * self.SurvivorRatio))
        else:
            print("Species:" + str(len(self.Species)))
            if len(self.Species) / self.PopulationSize > self.BalanceTop:
                self.SpeciesThreshold += self.BalanceFocus
                clean(self.Species)
                self.respeciate(self.PopulationSize - int(self.PopulationSize * self.SurvivorRatio))
            elif len(self.Species) / self.PopulationSize < self.BalanceBottom:
                self.SpeciesThreshold -= self.BalanceFocus
                clean(self.Species)
                self.respeciate(self.PopulationSize - int(self.PopulationSize * self.SurvivorRatio))
            else:
                clean(self.Species)
                self.speciate(self.PopulationSize - int(self.PopulationSize * self.SurvivorRatio))

    def respeciate(self, size):
        survivors = len(self.Population)
        self.Species = []
        for i in range(size):
            if random.random() < .5:
                child = self.Population[i % survivors].breed(random.choice(self.Population[:survivors]))
            else:
                child = self.Population[i % survivors].replicate()
            self.Population.append(child)
        for child in self.Population:
            self.add_to_specie(child)
        # if not len(self.Species) > self.PopulationSize:
        print("Threshold: " + str(self.SpeciesThreshold))
        print("Species:" + str(len(self.Species)))

    def speciate(self, size):
        survivors = len(self.Population)
        for i in range(size):
            child = self.Population[i % survivors].breed(random.choice(self.Population[:survivors]))
            self.Population.append(child)
            self.add_to_specie(child)
        # if not len(self.Species) > self.PopulationSize:
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
