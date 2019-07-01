from Drivers.NeatSpeciationDriver import NeatSpeciationDriver
from Nets.NeatNet import NeatNet
from Testers.EvolutionTrainer import EvolutionTrainer
from Nets.StaticEvolvingNet import StaticEvolvingNet
from Drivers.EvolutionSpeciationDriver import EvolutionSpeciationDriver
from Simulations.EvolutionSimulation import EvolutionSimulation


def IMITATOR(ar):
    return [1.0] if (ar[0] == 1.0) != (ar[1] == 1.0) else [0.0]


IN_SIZE = 2
OUT_SIZE = 1
LAYERS = 0
SIM = EvolutionSimulation(imitator=IMITATOR, in_dem=IN_SIZE, out_dem=OUT_SIZE, layers=LAYERS)

POPULATION_SIZE = 1000
SURVIVOR_RATIO = .5
GENERATION_SIZE = 4
SPECIES_THRESHOLD = 3.0
SISR = .1
BALANCING_FOCUS = 0.0
MUTABILITY = 50.0
EVOLVING_CLASS = NeatNet

Driver = NeatSpeciationDriver(population_size=POPULATION_SIZE,
                              reproducers_ratio=SURVIVOR_RATIO,
                              simulation=SIM,
                              generation_size=GENERATION_SIZE,
                              species_threshold=SPECIES_THRESHOLD,
                              balance_focus=BALANCING_FOCUS,
                              mutability_weights=2.0,
                              mutability_connections=.05,
                              mutability_nodes=.03,
                              mutability_reset=.1,
                              mutability_change_weight=.8,
                              mutability_toggle=.1,
                              excess_weight=1.0,
                              disjoint_weight=1.0,
                              weight_weight=0.4,
                              inter_species_breeding_rate=0.001,
                              asexual_breading_rate=.25
                              )

SIZE = [800, 800]
VERBOSITY = 2
ERROR_SIZE = 100
Trainer = EvolutionTrainer(driver=Driver,
                           size=SIZE,
                           verbosity=VERBOSITY,
                           error_length=ERROR_SIZE)

Trainer.main()
