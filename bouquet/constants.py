"""Constants and default configuration values for Bouquet"""

# Optimization defaults
DEFAULT_NUM_STEPS = 32
DEFAULT_INIT_STEPS = 5
DEFAULT_RELAXATION_STEPS = 50
DEFAULT_FMAX = 1e-3

# Auto-scaling thresholds for optimization steps based on dihedral count
AUTO_STEPS_THRESHOLDS = {
    3: 25,  # <= 3 dihedrals
    5: 50,  # <= 5 dihedrals
    7: 100,  # <= 7 dihedrals
}
AUTO_STEPS_DEFAULT = 200  # > 7 dihedrals

# Energy clipping for Bayesian optimization
ENERGY_CLIP_OFFSET = 2.0

# Gaussian process settings
GP_PERIOD_LENGTH_MEAN = 360
GP_PERIOD_LENGTH_STD = 0.1

# Acquisition function optimization
ACQ_NUM_RESTARTS = 64
ACQ_RAW_SAMPLES = 64

# Initial guess sampling
INITIAL_GUESS_STD = 90

# Supported energy/optimizer methods
SUPPORTED_METHODS = frozenset({"ani", "b3lyp", "b97", "gfn0", "gfn2", "gfnff"})

# Default methods
DEFAULT_ENERGY_METHOD = "gfn2"
DEFAULT_OPTIMIZER_METHOD = "gfnff"

# Psi4 settings
PSI4_NUM_THREADS = 4
