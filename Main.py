from Testers.EvolutionTrainer import EvolutionTrainer
from Nets.StaticEvolvingNet import StaticEvolvingNet
from Drivers.EvolutionSpeciationDriver import EvolutionSpeciationDriver
from Simulations.EvolutionSimulation import EvolveSimulation


def IMITATOR(ar):
    return ar


IN_SIZE = 2
OUT_SIZE = 2
LAYERS = 0
SIM = EvolveSimulation(imitator=IMITATOR, in_dem=IN_SIZE, out_dem=OUT_SIZE, layers = LAYERS)

POPULATION_SIZE = 300
SURVIVOR_RATIO = .75
GENERATION_SIZE = 4
SPECIES_THRESHOLD = 100.0
SISR = .01
BALANCING_FOCUS = 0.0
MUTABILITY = 50.0
EVOLVING_CLASS = StaticEvolvingNet

Driver = EvolutionSpeciationDriver(population_size=POPULATION_SIZE,
                                   survivor_ratio=SURVIVOR_RATIO,
                                   simulation=SIM,
                                   generation_size=GENERATION_SIZE,
                                   species_threshold=SPECIES_THRESHOLD,
                                   species_independent_survivor_ratio=SISR,
                                   balancing_focus=BALANCING_FOCUS,
                                   mutability=MUTABILITY,
                                   evolving_class=EVOLVING_CLASS)

SIZE = [500,500]
VERBOSITY = 2
ERROR_SIZE = 100
Trainer = EvolutionTrainer(driver=Driver,
                           size=SIZE,
                           verbosity=VERBOSITY,
                           error_length=ERROR_SIZE)

Trainer.main()