"""Physics checks with values that can be computed by hand."""

import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from qpmhc import core

PDB_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "1DUZ.pdb"
ORIGIN = np.array([[0.0, 0.0, 0.0]])


@lru_cache(maxsize=1)
def pyscf_can_run_hf():
    """True if a tiny PySCF RHF runs; checked in a subprocess because a broken
    native build (e.g. PySCF on Windows) can crash the interpreter outright."""
    code = ("from pyscf import gto, scf; "
            "scf.RHF(gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)).run()")
    try:
        return subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=120).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


requires_pyscf = pytest.mark.skipif(not pyscf_can_run_hf(), reason="PySCF missing or cannot run HF here")


def pair_energy(r, q1=0.0, q2=0.0):
    """Interaction energy of two atoms at distance r (A); returns (total, coul, lj)."""
    return core.interaction_energy(ORIGIN, np.array([q1]), np.array([[r, 0.0, 0.0]]), np.array([q2]))


def lj_only(r):
    """LJ energy (kcal/mol) of one uncharged pair at distance r (A)."""
    return pair_energy(r)[2]


# ------------------------------------------------------------------------------
# Coulomb with eps(r) = 4r
# ------------------------------------------------------------------------------
def test_coulomb_prefactor_two_unit_charges():
    # Two +1 e charges at 2 A: 332.06371 / (4 * 2 * 2) = 20.7539819 kcal/mol
    _, coul, _ = pair_energy(2.0, 1.0, 1.0)
    assert coul == pytest.approx(332.06371 / 16.0, rel=1e-12)
    assert coul == pytest.approx(20.7539819, rel=1e-8)


def test_coulomb_sign_follows_charges():
    assert pair_energy(3.0, 1.0, -1.0)[1] < 0.0
    assert pair_energy(3.0, -1.0, -1.0)[1] > 0.0


@pytest.mark.parametrize("r", [1.5, 2.0, 5.0, 10.0])
def test_coulomb_scales_as_inverse_r_squared(r):
    # eps(r) = 4r turns 1/(eps*r) into 1/(4 r^2): doubling r quarters E.
    e_r = pair_energy(r, 1.0, 1.0)[1]
    e_2r = pair_energy(2.0 * r, 1.0, 1.0)[1]
    assert e_r / e_2r == pytest.approx(4.0, rel=1e-12)


# ------------------------------------------------------------------------------
# Distance floor at R_MIN = 1 A
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("r", [0.0, 1e-6, 0.5, 0.999])
def test_distances_below_floor_use_floor(r):
    floored = pair_energy(r, 1.0, -1.0)
    at_floor = pair_energy(core.R_MIN, 1.0, -1.0)
    assert np.all(np.isfinite(floored))
    assert np.allclose(floored, at_floor, rtol=0.0, atol=0.0)


def test_floor_value_is_one_angstrom():
    assert core.R_MIN == 1.0


# ------------------------------------------------------------------------------
# Lennard-Jones
# ------------------------------------------------------------------------------
def test_lj_zero_at_sigma():
    assert lj_only(core.LJ_SIGMA) == pytest.approx(0.0, abs=1e-12)


def test_lj_minimum_is_minus_epsilon_at_r_min():
    r_star = 2.0 ** (1.0 / 6.0) * core.LJ_SIGMA
    assert lj_only(r_star) == pytest.approx(-core.LJ_EPSILON, rel=1e-12)
    # It is a minimum: a fine scan never goes below -epsilon, and neighbours are higher.
    scan = np.array([lj_only(r) for r in np.linspace(2.5, 6.0, 3501)])
    assert scan.min() >= -core.LJ_EPSILON - 1e-12
    assert lj_only(r_star - 0.01) > lj_only(r_star)
    assert lj_only(r_star + 0.01) > lj_only(r_star)


# ------------------------------------------------------------------------------
# Metropolis acceptance
# ------------------------------------------------------------------------------
def acceptance_frequency(monkeypatch, d_e, n_trials, seed=12345, temp=300.0):
    """Fraction of accepted one-step runs whose single trial move costs d_e (kcal/mol)."""
    calls = {"n": 0}

    def fake_energy(*_args):
        # Even calls are the starting energy (0), odd calls the trial (d_e).
        e = 0.0 if calls["n"] % 2 == 0 else d_e
        calls["n"] += 1
        return e, 0.0, 0.0

    monkeypatch.setattr(core, "interaction_energy", fake_energy)
    rng = np.random.default_rng(seed)
    accepted = 0
    for _ in range(n_trials):
        res = core.run_monte_carlo(ORIGIN, np.array([0.0]), ORIGIN + 5.0, np.array([0.0]),
                                   n_steps=1, temp=temp, rng=rng)
        accepted += res["acc_rate"] == 100.0
    return accepted / n_trials


@pytest.mark.parametrize("d_e", [-5.0, -0.1, 0.0])
def test_downhill_or_flat_moves_always_accepted(monkeypatch, d_e):
    assert acceptance_frequency(monkeypatch, d_e, n_trials=500) == 1.0


@pytest.mark.parametrize("d_e", [0.3, 0.6, 1.5])
def test_uphill_acceptance_matches_boltzmann_factor(monkeypatch, d_e):
    n = 20_000
    p = np.exp(-d_e / (core.KB * 300.0))
    sigma = np.sqrt(p * (1.0 - p) / n)          # binomial standard error
    freq = acceptance_frequency(monkeypatch, d_e, n_trials=n)
    assert abs(freq - p) < 4.0 * sigma


# ------------------------------------------------------------------------------
# Charge balance
# ------------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known issue, not fixed on purpose: the toy charge table is not neutral. "
        "For the heavy-atom 1DUZ peptide it sums to -3.489 e while the formal "
        "charge (and therefore the Mulliken sum in QM mode) is 0."
    ),
)
def test_standard_peptide_charge_matches_formal_charge():
    pep, _ = core.load_complex(str(PDB_PATH))
    toy_total = core.assign_toy_charges(pep).sum()
    assert toy_total == pytest.approx(core.peptide_formal_charge(pep), abs=0.05)


@requires_pyscf
def test_qm_mode_charges_sum_to_formal_charge():
    """QM mode is balanced by construction: Mulliken charges sum to the net charge."""
    water = {
        "elem": ["O", "H", "H"],
        "resname": ["HOH"] * 3,
        "name": ["O", "H1", "H2"],
        "xyz": np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]),
    }
    q, net = core.compute_qm_charges(water)
    assert net == 0
    assert q.sum() == pytest.approx(net, abs=1e-8)


@requires_pyscf
def test_mulliken_sum_equals_charge_of_a_cation():
    from pyscf import gto, scf

    d = 1.03 / np.sqrt(3.0)                     # NH4+, N-H = 1.03 A
    mol = gto.M(atom=f"N 0 0 0; H {d} {d} {d}; H {-d} {-d} {d}; H {-d} {d} {-d}; H {d} {-d} {-d}",
                      basis="3-21g", charge=1, verbose=0)
    mf = scf.RHF(mol).run()
    _, q = mf.mulliken_pop(verbose=0)
    assert q.sum() == pytest.approx(1.0, abs=1e-8)
