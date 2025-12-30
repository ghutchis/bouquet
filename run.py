import json
from argparse import ArgumentParser
from pathlib import Path
import hashlib
import logging
import sys

from datetime import datetime

import numpy as np
from ase.io.xyz import simple_write_xyz

from bouquet.setup import *
from bouquet.solver import run_optimization

logger = logging.getLogger('bouquet')

if __name__ == "__main__":
    # Parse the command line arguments
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int, default=datetime.now().microsecond,
                        help='Random seed')
    parser.add_argument('--smiles', type=str,
                        help='SMILES string of molecule to optimize')
    parser.add_argument('--file', type=str,
                        help='File containing the structure to optimize')
    parser.add_argument('--conformer-file', type=str,
                        help='File containing multiple conformers to use as initial guesses '
                             '(instead of random sampling). Overrides --init-steps.')
    parser.add_argument('--auto', action='store_true',
                        help='Set the number of steps based on the number of dihedrals')
    parser.add_argument('--num-steps', type=int, default=32,
                        help='Number of optimization steps to take')
    parser.add_argument('--init-steps', type=int, default=5,
                        help='Number of initial guesses to make (ignored if --conformer-file is provided)')
    parser.add_argument('--energy', choices=['ani', 'b3lyp', 'b97',
                        'gfn0', 'gfn2', 'gfnff'], default='gfn2', help='Energy method')
    parser.add_argument('--optimizer', choices=['ani', 'b3lyp', 'b97',
                        'gfn0', 'gfn2', 'gfnff'], default='gfnff', help='Optimizer method')
    parser.add_argument('--relax', action='store_true',
                        help='Relax the non-dihedral degrees of freedom before computing energy')
    args = parser.parse_args()

    if args.smiles is None and args.file is None:
        raise ValueError('Must specify either --smiles or --file')

    if args.smiles is not None:
        name = args.smiles
    else:
        name = Path(args.file).stem

    # Make an output directory
    params_hash = hashlib.sha256(str(args.__dict__).encode()).hexdigest()
    out_dir = Path(__file__).parent.joinpath(
        f'solutions/{name}-{args.seed}-{args.energy}-{params_hash[-6:]}')
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_dir.joinpath('run_params.json').open('w') as fp:
        json.dump(args.__dict__, fp)

    # Set up the logging
    handlers = [logging.FileHandler(out_dir.joinpath('runtime.log')),
                logging.StreamHandler(sys.stdout)]

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO, handlers=handlers)
    logger.info(f'Started optimizing the conformers for {name}')

    # Make the initial guess
    if args.file is None:
        # this will do some initial cleanup from the SMILES string
        init_atoms, mol = get_initial_structure(args.smiles)
    else:
        # this will just read the geometry from the file
        # and parse using Pybel
        init_atoms, mol = get_initial_structure_from_file(args.file)
    logger.info(f'Determined initial structure with {len(init_atoms)} atoms')

    # TODO: have optional cleanups

    # Detect the dihedral angles
    dihedrals = detect_dihedrals(mol)
    logger.info(f'Detected {len(dihedrals)} dihedral angles')

    # Load conformers if provided
    initial_conformers = None
    if args.conformer_file is not None:
        initial_conformers = get_conformers_from_file(args.conformer_file)
        logger.info(f'Loaded {len(initial_conformers)} conformers from {args.conformer_file}')
        # Validate that conformers have the same number of atoms
        for i, conf in enumerate(initial_conformers):
            if len(conf) != len(init_atoms):
                raise ValueError(f'Conformer {i} has {len(conf)} atoms, expected {len(init_atoms)}')

    # check if we should auto-set the number of steps
    if args.auto:
        if len(dihedrals) <= 3:
            # 25 total counting the initial random choices/conformers
            if initial_conformers is not None:
                args.num_steps = max(0, 25 - len(initial_conformers))
            else:
                args.num_steps = 25 - args.init_steps
        elif len(dihedrals) <= 5:
            # 50 total counting the initial random choices/conformers
            if initial_conformers is not None:
                args.num_steps = max(0, 50 - len(initial_conformers))
            else:
                args.num_steps = 50 - args.init_steps
        elif len(dihedrals) <= 7:
            # 100 total counting the initial random choices/conformers
            if initial_conformers is not None:
                args.num_steps = max(0, 100 - len(initial_conformers))
            else:
                args.num_steps = 100 - args.init_steps
        else:
            # 200 total counting the initial random choices/conformers
            if initial_conformers is not None:
                args.num_steps = max(0, 200 - len(initial_conformers))
            else:
                args.num_steps = 200 - args.init_steps

    # Save the initial guess
    with out_dir.joinpath('initial.xyz').open('w') as fp:
        simple_write_xyz(fp, [init_atoms])

    # Set up the optimization problem
    if args.energy == 'ani':
        import torchani
        calc = torchani.models.ANI2x().ase()
    elif args.energy == 'xtb' or args.energy == 'gfn2':
        from xtb.ase.calculator import XTB
        calc = XTB()  # gfn2
    elif args.energy == 'gfn0':
        from xtb.ase.calculator import XTB
        calc = XTB(method='gfn0')
    elif args.energy == 'gfnff':
        from xtb.ase.calculator import XTB
        calc = XTB(method='gfnff')
    elif args.energy == 'b3lyp':
        from ase.calculators.psi4 import Psi4
        calc = Psi4(method='b3lyp-D3MBJ2B', basis='def2-svp',
                    num_threads=4, multiplicity=1, charge=0)
    elif args.energy == 'b97':
        from ase.calculators.psi4 import Psi4
        calc = Psi4(method='b97-d3bj', basis='def2-svp',
                    num_threads=4, multiplicity=1, charge=0)
    else:
        raise ValueError(f'Unrecognized QC method: {args.energy}')

    # default to using the same method for relaxation
    relaxCalc = calc
    if args.optimizer == 'ani':
        import torchani
        relaxCalc = torchani.models.ANI2x().ase()
    elif args.optimizer == 'xtb' or args.optimizer == 'gfn2':
        from xtb.ase.calculator import XTB
        relaxCalc = XTB()  # gfn2
    elif args.optimizer == 'gfn0':
        from xtb.ase.calculator import XTB
        relaxCalc = XTB(method='gfn0')
    elif args.optimizer == 'gfnff':
        from xtb.ase.calculator import XTB
        relaxCalc = XTB(method='gfnff')
    elif args.optimizer == 'b3lyp':
        from ase.calculators.psi4 import Psi4
        relaxCalc = Psi4(method='b3lyp-D3MBJ2B', basis='def2-svp',
                    num_threads=4, multiplicity=1, charge=0)
    elif args.optimizer == 'b97':
        from ase.calculators.psi4 import Psi4
        relaxCalc = Psi4(method='b97-d3bj', basis='def2-svp',
                    num_threads=4, multiplicity=1, charge=0)
    else:
        raise ValueError(f'Unrecognized QC method: {args.optimizer}')

    final_atoms = run_optimization(init_atoms, dihedrals, args.num_steps, calc, relaxCalc,
                                   args.init_steps, out_dir, relax=args.relax, seed=args.seed,
                                   initial_conformers=initial_conformers)

    # Save the final structure
    with out_dir.joinpath('final.xyz').open('w') as fp:
        simple_write_xyz(fp, [final_atoms])
    logger.info(f'Done. Files are stored in {str(out_dir)}')
