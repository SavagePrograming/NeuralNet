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

print(Net.in_dem)
print(Net.out_dem)
print(Net.connection_genes)

print(NEW.in_dem)
print(NEW.out_dem)
print(NEW.connection_genes)
