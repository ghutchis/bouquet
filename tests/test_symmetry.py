"""Tests for exact graph-symmetry rotor periodicity across setup, surrogate, and ensemble."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem

from bouquet.calc_rdkit import RDKitMMFFCalculator
from bouquet.ensemble import _diversity_factor, _periodic_min_dist
from bouquet.priors import DihedralPriorModule
from bouquet.setup import detect_dihedrals, mol_to_ase_atoms
from bouquet.solver import plan_initial_points
from bouquet.subspace import LowEnergySubspace
from bouquet.surrogate import (
    _SymmetryScaledPrior,
    _fit_gradient_gp,
    _select_next_points_botorch,
)


def _orders_by_bond(smiles: str) -> dict[tuple[str, str], list[int]]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    dihs = detect_dihedrals(mol)
    out: dict[tuple[str, str], list[int]] = {}
    for d in dihs:
        u, v = d.chain[1], d.chain[2]
        key = tuple(sorted((mol.GetAtomWithIdx(u).GetSymbol(), mol.GetAtomWithIdx(v).GetSymbol())))
        out.setdefault(key, []).append(d.symmetry_order)
    return out


def _orders(smiles: str) -> list[int]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    return sorted(d.symmetry_order for d in detect_dihedrals(mol))


class TestSymmetryDetection:
    def test_c3_terminal_tops(self):
        # Butane: two terminal -CH3 (C3) + one central C-C (C1)
        assert _orders("CCCC") == [1, 3, 3]
        # Ethylammonium: -CH3 (C3) + -NH3+ (C3)
        assert _orders("CC[NH3+]") == [3, 3]
        # Trifluoroethane: -CH3 (C3) and -CF3 (C3) on the same bond -> lcm(3, 3) = 3
        assert _orders("CC(F)(F)F") == [3]

    def test_c2_planar_and_aromatic_rotors(self):
        # Biphenyl: both sides C2 -> lcm(2, 2) = 2
        assert _orders("c1ccccc1-c2ccccc2") == [2]
        # para-Cl-biphenyl: both sides C2 -> 2
        assert _orders("Clc1ccc(-c2ccccc2)cc1") == [2]
        # para-CN-phenyl-ethane: -CH3 (3), Et-Ph (2)
        assert _orders("CCc1ccc(C#N)cc1") == [2, 3]
        # 9-anthryl-ethane: -CH3 (3), Et-anthryl (2)
        assert _orders("CCc1c2ccccc2cc2ccccc12") == [2, 3]
        # 4-pyridyl-phenyl: both sides C2 -> 2
        assert _orders("c1ccc(-c2ccncc2)cc1") == [2]
        # Nitrobenzene & benzoate: -NO2 / -CO2- (2) + Ph (2) -> 2
        assert _orders("O=[N+]([O-])c1ccccc1") == [2]
        assert _orders("O=C([O-])c1ccccc1") == [2]

    def test_c6_combined_c2_and_c3(self):
        assert _orders("Cc1ccccc1") == [6]          # toluene
        assert _orders("FC(F)(F)c1ccccc1") == [6]   # benzotrifluoride
        assert _orders("[n+]1(ccccc1)C") == [6]     # N-methylpyridinium

    def test_negative_controls_stay_order_1(self):
        # meta-Cl-phenyl & 2-naphthyl & 2-pyridyl are NOT C2 symmetric
        assert _orders("Clc1cccc(-c2ccccc2)c1") == [2]  # unsubstituted Ph side is still C2!
        assert _orders("CCc1cccc(Cl)c1") == [1, 3]      # m-Cl-Ph is 1, -CH3 is 3
        assert _orders("CCc1ccc2ccccc2c1") == [1, 3]    # 2-naphthyl is 1, -CH3 is 3
        assert _orders("CCc1ccccn1") == [1, 3]          # 2-pyridyl is 1, -CH3 is 3
        # para-OH and para-Me have internal rotatable bonds -> phenyl bond stays 1
        assert _orders("CCc1ccc(O)cc1") == [1, 1, 3]
        assert _orders("CCc1ccc(C)cc1") == [1, 3, 3]
        # Neutral cysteine: pyramidal -NH2 must stay 1; -COOH has -OH rotor -> 1
        assert _orders("C([C@@H](C(=O)O)N)S") == [1, 1, 1, 1, 1]
        # Zwitterionic cysteine: -CO2- is 2, -NH3+ is 3
        assert _orders("C([C@@H](C(=O)[O-])[NH3+])S") == [1, 1, 2, 3]
        # Neutral aniline: pyramidal-prone -NH2 stays 1 (phenyl side is C2 -> 2)
        assert _orders("Nc1ccccc1") == [2]
        # t-Bu-benzene: 3 methyls (3), stem bond has C2 from phenyl side -> [2, 3, 3, 3]
        assert _orders("CC(C)(C)c1ccccc1") == [2, 3, 3, 3]
        # Chiral center next to phenyl: phenyl flip is STILL exact C2
        assert _orders("C[C@H](O)c1ccccc1") == [1, 2, 3]

    @pytest.mark.parametrize(
        "smiles",
        [
            "CCCC",
            "Cc1ccccc1",
            "C[C@H](O)c1ccccc1",
            "O=[N+]([O-])c1ccccc1",
            "O=C([O-])c1ccccc1",
            "C([C@@H](C(=O)[O-])[NH3+])S",
        ],
    )
    def test_detected_symmetries_preserve_mmff_energy(self, smiles):
        """Rotating any symmetric dihedral by 360/m leaves the relaxed energy invariant."""
        from bouquet.assess import relax_structure

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        params = AllChem.ETKDGv3()
        params.randomSeed = 7
        AllChem.EmbedMolecule(mol, params)
        AllChem.MMFFOptimizeMolecule(mol)
        dihs = detect_dihedrals(mol)
        calc = RDKitMMFFCalculator(mol)
        atoms = mol_to_ase_atoms(mol)
        e0, atoms = relax_structure(atoms, calc, calc, steps=None)

        sym_dihs = [d for d in dihs if d.symmetry_order > 1]
        assert sym_dihs
        for d in sym_dihs:
            rot = atoms.copy()
            rot.set_dihedral(*d.chain, d.get_angle(rot) + d.period_deg, indices=d.group)
            e1, _ = relax_structure(rot, calc, calc, steps=None)
            assert abs(e1 - e0) < 1e-4


class TestSymmetryInSurrogateAndEnsemble:
    def test_select_next_points_stays_in_irreducible_domain(self):
        """Proposals for symmetric rotors stay inside [0, 360 / m_j)."""
        torch.manual_seed(0)
        sym = (1, 2, 3)
        X = torch.tensor(
            [[10.0, 20.0, 30.0], [180.0, 90.0, 60.0], [270.0, 150.0, 110.0], [50.0, 10.0, 5.0]],
            dtype=torch.float64,
        )
        y = torch.tensor([0.0, -0.2, 0.3, -0.1], dtype=torch.float64)
        cand = _select_next_points_botorch(
            X, y, acq_num_restarts=4, acq_raw_samples=8, symmetry_orders=sym
        )
        assert 0.0 <= cand[0] <= 360.0
        assert 0.0 <= cand[1] <= 180.0
        assert 0.0 <= cand[2] <= 120.0

    def test_gradient_gp_chain_rule_with_symmetry_order(self):
        """Gradient GP with m=3 matches exact derivative scaling for E(theta) = -cos(3*theta)."""
        # E(theta) = -cos(3*theta_rad), dE/dtheta_rad = 3*sin(3*theta_rad), period 120 deg
        deg = torch.linspace(5.0, 115.0, 8, dtype=torch.float64).unsqueeze(-1)
        rad = deg * (math.pi / 180.0)
        raw_y = -torch.cos(3.0 * rad).squeeze(-1)
        grads = (3.0 * torch.sin(3.0 * rad))  # (8, 1) eV/rad

        sym = (3,)
        periods = torch.tensor([120.0], dtype=torch.float64)
        train_x = torch.remainder(deg, periods) / periods
        neg = (-raw_y)[:, None]
        y_std = neg.std(dim=0, keepdim=True)
        neg_mean = neg.mean(dim=0, keepdim=True)
        train_y = (neg - neg_mean) / y_std
        energy_cap = torch.full_like(raw_y, 10.0)

        gp = _fit_gradient_gp(
            train_x, train_y, raw_y, energy_cap, grads, y_std, symmetry_orders=sym
        )
        # Predict at 60 deg (= u=0.5, true E = -cos(pi) = +1.0) vs 180 deg (equivalent!)
        u_test = torch.tensor([[0.5], [0.0]], dtype=torch.float64).unsqueeze(1)
        with torch.no_grad():
            mu_std = gp.posterior(u_test).mean.reshape(-1)
        e_pred = -(mu_std * y_std.item() + neg_mean.item())
        assert float(e_pred[0]) == pytest.approx(1.0, abs=0.05)   # 60 deg -> +1.0
        assert float(e_pred[1]) == pytest.approx(-1.0, abs=0.05)  # 0 deg -> -1.0

    def test_symmetry_scaled_prior_matches_true_angle(self):
        pm = DihedralPriorModule(
            dim=2,
            univariate_assignments={0: "sp3_sp3", 1: "sp3_sp2"},
            bivariate_assignments={},
            univariate_priors={},
            bivariate_priors={},
            input_in_degrees=False,
        )
        wrapped = _SymmetryScaledPrior(pm, (3, 2))
        # u = (0.5, 0.5) in symmetry-folded space = (60 deg, 90 deg) = (1/6, 1/4) in full turn
        u = torch.tensor([[[0.5, 0.5]]], dtype=torch.float64)
        x_full = torch.tensor([[[60.0 / 360.0, 90.0 / 360.0]]], dtype=torch.float64)
        assert torch.allclose(wrapped(u), pm(x_full), atol=1e-12)

    def test_ensemble_periodic_distance_folds_symmetric_rotors(self):
        periods = torch.tensor([180.0, 120.0], dtype=torch.float64)
        minima = torch.tensor([[10.0, 20.0]], dtype=torch.float64)
        # (190, 140) is symmetry-equivalent to (10, 20) under (C2, C3) -> distance 0
        equiv = torch.tensor([190.0, 140.0], dtype=torch.float64)
        assert float(_periodic_min_dist(equiv, minima, periods_deg=periods)) == pytest.approx(0.0, abs=1e-12)
        # _diversity_factor suppresses at u = (10/180, 20/120)
        u_at = torch.tensor([[[10.0 / 180.0, 20.0 / 120.0]]], dtype=torch.float64)
        assert float(_diversity_factor(u_at, minima, 0.5, periods_deg=periods)) == pytest.approx(0.5, abs=1e-9)

    def test_plan_initial_points_dedups_symmetric_peaks(self):
        pm = DihedralPriorModule(
            dim=1,
            univariate_assignments={0: "sp3_sp3"},  # peaks at 60, 180, 300
            bivariate_assignments={},
            univariate_priors={},
            bivariate_priors={},
        )
        pts = plan_initial_points(
            pm, 1, np.array([0.0]), init_steps=5, grid_budget=64, seed=0,
            symmetry_orders=(3,),
        )
        # 60, 180, 300 all fold to 60 mod 120 -> single unique peak!
        assert len(pts) == 1
        assert float(pts[0, 0]) == pytest.approx(60.0)

    def test_low_energy_subspace_dedups_symmetric_copies(self):
        ss = LowEnergySubspace(n_dihedrals=2, dedup_thresh_deg=15.0, symmetry_orders=(2, 3))
        coords = np.array([[10.0, 20.0], [190.0, 140.0], [90.0, 80.0]])
        energies = np.array([0.0, 0.01, 0.02])
        ss.update(coords, energies)
        # Row 1 is symmetry-equivalent to row 0 -> only 2 elite conformers kept
        assert ss.n_elite == 2
