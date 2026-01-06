"""Constants and default configuration values for Bouquet"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional


# Type alias for supported methods
MethodType = Literal["ani", "b3lyp", "b97", "gfn0", "gfn2", "gfnff"]


@dataclass
class Configuration:
    """Configuration for a Bouquet optimization run."""

    # Input specification (one of these must be provided)
    smiles: Optional[str] = None
    input_file: Optional[Path] = None
    conformer_file: Optional[Path] = None

    # Output name
    name: Optional[str] = None

    # Calculation methods
    energy_method: MethodType = "gfn2"
    optimizer_method: MethodType = "gfnff"

    # Optimization parameters
    num_steps: int = 32
    init_steps: int = 5
    auto_steps: bool = False
    relax: bool = True
    seed: int = field(default_factory=lambda: datetime.now().microsecond)

    # Output
    out_dir: Optional[Path] = None

    # Psi4 settings (for DFT methods)
    num_threads: int = 4
    charge: int = 0
    multiplicity: int = 1

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.smiles is None and self.input_file is None:
            raise ValueError("Must specify either smiles or input_file")

        # Convert string paths to Path objects if needed
        if isinstance(self.input_file, str):
            self.input_file = Path(self.input_file)
        if isinstance(self.conformer_file, str):
            self.conformer_file = Path(self.conformer_file)
        if isinstance(self.out_dir, str):
            self.out_dir = Path(self.out_dir)

    def compute_auto_steps(self, num_dihedrals: int, num_initial: int) -> int:
        """Compute the number of optimization steps based on dihedral count.

        Args:
            num_dihedrals: Number of dihedral angles detected
            num_initial: Number of initial conformers/guesses

        Returns:
            Number of optimization steps to perform
        """
        if not self.auto_steps:
            return self.num_steps

        total = AUTO_STEPS_DEFAULT
        for threshold, steps in sorted(AUTO_STEPS_THRESHOLDS.items()):
            if num_dihedrals <= threshold:
                total = steps
                break

        return max(0, total - num_initial)


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
NUM_THREADS = 4
