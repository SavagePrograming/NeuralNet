import random

import pygame

from Nets.NeatNet import NeatNet
from SupportClasses.GeneticsPackage import GeneticsPackage

IN_DEM = 2
OUT_DEM = 4
CONNECTION_GENES = []

gene_controller = GeneticsPackage(IN_DEM, OUT_DEM)
Net = NeatNet(IN_DEM, OUT_DEM, CONNECTION_GENES, gene_controller)
Net.add_connection_random(0, -1, CONNECTION_GENES)
Net = NeatNet(IN_DEM, OUT_DEM, CONNECTION_GENES, gene_controller)

save = Net.save()
NEW = NeatNet(0, 0, [], gene_controller)
NEW.load(save)

pygame.init()
Screen = pygame.display.set_mode([400, 400])
pygame.key.set_repeat(100, 50)
Screen.fill([0, 0, 100])
# print(NEW.weights)
genes = NEW.connection_genes.copy()
# print(genes)
NEW.add_node(0, genes)
# print(genes)
# print(NEW.in_dem)
NEW = NeatNet(NEW.in_dem - 1, NEW.out_dem, genes, gene_controller)
# print(NEW.in_dem)
# print(NEW.weights)
# print(Net.weights)

Net.update(Screen, 0, 0, 400, 400)
NEW.update(Screen, 0, 0, 400, 400)

count = 0
while True:
    Screen.fill([0, 0, 100])
    NEW.draw()
    pygame.event.get()
    pygame.display.flip()
    count += 1
    if count % 1000 == 0:
        genes = NEW.connection_genes.copy()
        OUT_LIST = list(range(-1, -OUT_DEM - 1, -1)) + list(range(IN_DEM+1, gene_controller.node_innovation_number))
        # print(OUT_LIST)
        NEW.add_connection_random(random.randint(0, gene_controller.node_innovation_number - 1), random.choice(OUT_LIST), genes)
        NEW.add_node(random.randint(0, len(genes)-1), genes)
        NEW = NeatNet(NEW.in_dem - 1, NEW.out_dem, genes, gene_controller)
        NEW.update(Screen, 0, 0, 400, 400)


# print(Net.in_dem)
# print(Net.out_dem)
# print(Net.connection_genes)
#
# print(NEW.in_dem)
# print(NEW.out_dem)
# print(NEW.connection_genes)
