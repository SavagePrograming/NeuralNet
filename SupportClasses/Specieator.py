from typing import List, Tuple

import math
import pygame

from Nets import NeatNet
from formulas import not_none


class Specieator:
    def __init__(self, species_threshold, min_champion_size, balance_top, balance_bottom, balance_focus,
                 stagnant_generations):
        self.stagnant_generations = stagnant_generations
        self.balance_focus = balance_focus
        self.balance_bottom = balance_bottom
        self.balance_top = balance_top
        self.species: List[List[NeatNet]] = []
        self.species_score: List[Tuple[int, int]] = []
        self.species_threshold: float = species_threshold
        self.min_champion_size = min_champion_size

    def score_species(self, generation_count):
        for i in range(len(self.species)):
            score = max(self.species[i]).score
            if score > self.species_score[i][0]:
                self.species_score[i] = (score, generation_count)

    def add_to_specie(self, child):
        min_dist = self.species_threshold
        min_species = -1
        for s in range(len(self.species)):
            if child.distance(self.species[s][0]) < min_dist:
                min_dist = child.distance(self.species[s][0])
                min_species = s
        if min_species == -1:
            child.specie = len(self.species)
            self.species.append([child])
            self.species_score.append((0, self.generation_count))
        else:
            child.specie = min_species
            self.species[min_species].append(child)

    def clear(self):
        self.species = []
        self.species_score = []

    def clean(self):
        i = 0
        while i < len(self.species):
            if not self.species[i]:
                del self.species[i]
                del self.species_score[i]
            else:
                i += 1

    def add_all_to_species(self, array, generation_count):
        self.generation_count = generation_count
        list(map(self.add_to_specie, array))

    def get_species(self, net):
        for i in range(len(self.species)):
            if net in self.species[i]:
                return i
        return -1

    def get_species_champions(self):
        return list(map(max, filter(self.species_filter, self.species)))

    def species_filter(self, species):
        return len(species) > self.min_champion_size

    def remove_all_but(self, keep):
        self.keep = keep
        self.species = list(map(self.remove_all_from_one_specie, self.species))

    def remove_all_from_one_specie(self, specie):
        return list(filter(self.remove_child_filter, specie))

    def remove_child_filter(self, child):
        return child in self.keep

    def adjust_scores(self):
        for s in range(len(self.species)):
            size = len(self.species[s])
            for net in self.species[s]:
                net.score /= size

    def balance_threshold(self, population_size) -> bool:
        if self.balance_focus == 0.0:
            return False
        elif len(self.species) / population_size > self.balance_top:
            self.species_threshold += self.balance_focus
            return True
        elif len(self.species) / population_size < self.balance_bottom:
            self.species_threshold -= self.balance_focus
            return True
        return False

    def remove_stagnant_species(self, generation_count):
        # print("--------------------")
        self.generation_count = generation_count
        removables = list(filter(not_none, map(self.mark_stagnant, range(len(self.species)))))
        removables2 = list(filter(not_none, map(self.mark_stagnant_score, range(len(self.species)))))
        # print("====================")
        list(map(self.species.remove, removables))
        list(map(self.species_score.remove, removables2))
        return removables


    def mark_stagnant(self, i):
        if self.generation_count - self.species_score[i][1] >= self.stagnant_generations:
            # print(len(self.species))
            # print(i)
            return self.species[i]
        return None
    def mark_stagnant_score(self, i):
        if self.generation_count - self.species_score[i][1] >= self.stagnant_generations:
            return self.species_score[i]
        return None


    def draw(self, screen: pygame.Surface, x: int, y: int, width: int, height: int, dot_size: int = 10):

        self.row_count = int(math.sqrt(len(self.species))) + 1
        self.row_size = math.ceil(len(self.species) / self.row_count)

        for i in range(self.row_count * self.row_size):
            if (i < len(self.species)):
                self.species[i][0].update(screen,
                                          x + (i % self.row_size) * (width // self.row_size),
                                          y + (i // self.row_size) * (height // self.row_count),
                                          width // self.row_size,
                                          height // self.row_count,
                                          dot_size)
                self.species[i][0].draw()
