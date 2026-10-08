"""
Q-pMHC
======
A deliberately minimal, didactic rigid-body Metropolis sampler for a
peptide-MHC complex, built to explore one question:

    Does a quantum-derived (HF/Mulliken) peptide charge distribution give a
    different peptide-MHC electrostatic interaction than fixed force-field
    charges?

It combines: PDB parsing (Biopython) -> partial charges (toy AMBER-like table
or PySCF Hartree-Fock) -> Coulomb + LJ interaction energy -> Metropolis Monte
Carlo of rigid-body moves.

THIS IS A TOY. KNOWN LIMITATIONS (on purpose, to keep the code readable):
  1. Classical charges are a tiny hand-written table (backbone + formal charges
     on charged side chains), not a real force field (use AmberTools/OpenMM).
     The table is not neutral by construction: its per-residue sum is not the
     residue's formal charge, and the termini get no charge.
  2. Electrostatics use a distance-dependent dielectric (eps = 4r) as a crude
     stand-in for solvent screening. No Generalized Born / Poisson-Boltzmann.
  3. LJ is one generic (sigma, epsilon) for every atom pair.
  4. QM charges: gas-phase HF/small basis, computed once at the starting
     geometry, NO electrostatic embedding (the peptide is not polarized by the
     MHC). Mulliken charges are strongly basis-set dependent.
  5. Sampling is rigid-body only (translation + rotation). No peptide or
     side-chain flexibility, no minimization. Uncertainty is only a
     block-averaging SEM (plus optional seed-to-seed spread).
  6. The output is an interaction energy, NOT a binding free energy: no
     desolvation, entropy or protein reorganization.
  7. Starting from a crystal structure, most moves raise the energy: the
     sampler mostly measures the local "stiffness" of the toy potential.

Usage (the default mode is qm, which needs hydrogens):
    qpmhc data/raw/1DUZ.pdb --mode standard
    qpmhc data/raw/1DUZ.pdb --mode qm --add-hydrogens
"""

import argparse
import os
import tempfile

import numpy as np
from Bio.PDB import PDBParser
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

# ------------------------------------------------------------------------------
# 1. Constants and toy-model parameters
# ------------------------------------------------------------------------------
KB = 0.0019872042            # Boltzmann constant, kcal/(mol*K)
COULOMB_CONST = 332.06371    # kcal*A/(mol*e^2)
EPS_SLOPE = 4.0              # distance-dependent dielectric: eps(r) = EPS_SLOPE * r
R_MIN = 1.0                  # A, soft floor on distances (avoids singularities)
LJ_SIGMA = 3.0               # A, generic for every pair (toy)
LJ_EPSILON = 0.1             # kcal/mol, generic for every pair (toy)
POCKET_CUTOFF = 8.0          # A, receptor atoms kept around the peptide

# Backbone partial charges (AMBER-like values, kept from the original script)
BACKBONE_CHARGES = {
    "N": -0.4157, "H": 0.2719, "CA": -0.0014, "HA": 0.0823,
    "C": 0.5973, "O": -0.5679,
}
# Crude formal charges on charged side chains, keyed by (residue, atom)
SIDECHAIN_CHARGES = {
    ("LYS", "NZ"): 1.0,
    ("ARG", "CZ"): 1.0,
    ("ASP", "OD1"): -0.5, ("ASP", "OD2"): -0.5,
    ("GLU", "OE1"): -0.5, ("GLU", "OE2"): -0.5,
}
RESIDUE_FORMAL_CHARGE = {"LYS": 1, "ARG": 1, "ASP": -1, "GLU": -1}
ATOMIC_NUMBER = {"H": 1, "C": 6, "N": 7, "O": 8, "S": 16}


# ------------------------------------------------------------------------------
# 2. Structure handling
# ------------------------------------------------------------------------------
def add_hydrogens(pdb_path, ph=7.0):
    """Protonate with PDBFixer (optional dependency). Returns a temp PDB path."""
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    fixer = PDBFixer(filename=pdb_path)
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.missingResidues = {}          # do not model missing loops in a toy
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    out = os.path.join(tempfile.mkdtemp(), "protonated.pdb")
    with open(out, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh)
    return out


def toy_charge(resname, atom_name):
    """Toy classical charge for one atom (0.0 if not in the table)."""
    if (resname, atom_name) in SIDECHAIN_CHARGES:
        return SIDECHAIN_CHARGES[(resname, atom_name)]
    return BACKBONE_CHARGES.get(atom_name, 0.0)


def load_complex(pdb_path, receptor_chain="A", peptide_chain="C", keep_h=False):
    """
    Returns (peptide, receptor) dicts with arrays: xyz, elem, resname, name.
    Receptor is restricted to atoms within POCKET_CUTOFF of the peptide.
    Waters/heteroatoms are ignored.
    """
    structure = PDBParser(QUIET=True).get_structure("MHC", pdb_path)
    model = structure[0]                # a Structure holds models, not chains

    def collect(chain):
        rows = []
        for res in chain:
            if res.id[0] != " ":        # skip waters / ligands
                continue
            for atom in res:
                elem = (atom.element or atom.get_name()[0]).strip().upper()
                if elem == "H" and not keep_h:
                    continue
                rows.append((atom.get_coord(), elem, res.get_resname(), atom.get_name()))
        return {
            "xyz": np.array([r[0] for r in rows]),
            "elem": [r[1] for r in rows],
            "resname": [r[2] for r in rows],
            "name": [r[3] for r in rows],
        }

    pep = collect(model[peptide_chain])
    rec = collect(model[receptor_chain])

    near = cdist(rec["xyz"], pep["xyz"]).min(axis=1) < POCKET_CUTOFF
    rec = {k: (v[near] if k == "xyz" else [x for x, keep in zip(v, near) if keep])
           for k, v in rec.items()}
    return pep, rec


def assign_toy_charges(atoms):
    return np.array([toy_charge(r, n) for r, n in zip(atoms["resname"], atoms["name"])])


# ------------------------------------------------------------------------------
# 3. Quantum engine (PySCF Hartree-Fock)
# ------------------------------------------------------------------------------
def peptide_formal_charge(pep):
    """Net charge at pH ~7 from charged side chains (LYS/ARG +1, ASP/GLU -1).

    Zwitterionic termini (+1 and -1) are assumed and cancel; HIS is neutral.
    """
    residues = []                       # one entry per residue, via its CA atom
    for r, n in zip(pep["resname"], pep["name"]):
        if n == "CA":
            residues.append(r)
    return sum(RESIDUE_FORMAL_CHARGE.get(r, 0) for r in residues)


def compute_qm_charges(pep, basis="3-21g"):
    """HF/basis Mulliken charges of the isolated peptide (gas phase, no embedding)."""
    from pyscf import gto, scf

    if "H" not in pep["elem"]:
        raise ValueError("QM mode needs hydrogens. Re-run with --add-hydrogens.")

    charge = peptide_formal_charge(pep)
    n_electrons = sum(ATOMIC_NUMBER[e] for e in pep["elem"]) - charge
    if n_electrons % 2:
        raise ValueError(
            f"Odd electron count ({n_electrons}) for net charge {charge:+d}: "
            "check protonation/termini. Closed-shell RHF is not valid here."
        )

    atom_spec = [[e, tuple(x)] for e, x in zip(pep["elem"], pep["xyz"])]
    mol = gto.M(atom=atom_spec, basis=basis, spin=0, charge=charge, verbose=0)
    mf = scf.RHF(mol).run()
    if not mf.converged:
        print("    WARNING: SCF did not converge; charges are unreliable.")
    _, qm_charges = mf.mulliken_pop(verbose=0)
    return np.asarray(qm_charges), charge


# ------------------------------------------------------------------------------
# 4. Interaction energy: screened Coulomb + generic Lennard-Jones
# ------------------------------------------------------------------------------
def interaction_energy(pep_xyz, pep_q, rec_xyz, rec_q):
    """
    E_coul = 332.06 * sum q_i q_j / (eps(r) * r),  eps(r) = EPS_SLOPE * r
           = 332.06 * sum q_i q_j / (EPS_SLOPE * r^2)
    E_lj   = sum 4*eps*((s/r)^12 - (s/r)^6)
    Distances are floored at R_MIN instead of dropping pairs, so the energy
    stays continuous.
    """
    r = np.maximum(cdist(pep_xyz, rec_xyz), R_MIN)
    e_coul = COULOMB_CONST * np.sum(np.outer(pep_q, rec_q) / (EPS_SLOPE * r**2))
    sr6 = (LJ_SIGMA / r) ** 6
    e_lj = np.sum(4.0 * LJ_EPSILON * (sr6**2 - sr6))
    return e_coul + e_lj, e_coul, e_lj


# ------------------------------------------------------------------------------
# 5. Metropolis Monte Carlo (rigid-body moves)
# ------------------------------------------------------------------------------
def run_monte_carlo(pep_xyz, pep_q, rec_xyz, rec_q, n_steps=2000, temp=300.0,
                    max_shift=0.3, max_angle_deg=3.0, burn_in=0.2, rng=None):
    """
    Random rigid translation (+/- max_shift A per axis) and rotation about the
    peptide centroid (random axis, angle in +/- max_angle_deg), accepted with
    P = min(1, exp(-dE / kB T)). Proposals are symmetric, so plain Metropolis.
    """
    rng = rng or np.random.default_rng()
    coords = pep_xyz.copy()
    e_cur, _, _ = interaction_energy(coords, pep_q, rec_xyz, rec_q)
    e_start = e_cur
    trace, accepted = [e_cur], 0

    for _ in range(n_steps):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = np.deg2rad(rng.uniform(-max_angle_deg, max_angle_deg))
        rot = Rotation.from_rotvec(axis * angle)
        centroid = coords.mean(axis=0)
        trial = rot.apply(coords - centroid) + centroid + rng.uniform(-max_shift, max_shift, 3)

        e_trial, _, _ = interaction_energy(trial, pep_q, rec_xyz, rec_q)
        d_e = e_trial - e_cur
        if d_e <= 0 or rng.random() < np.exp(-d_e / (KB * temp)):
            coords, e_cur = trial, e_trial
            accepted += 1
        trace.append(e_cur)

    trace = np.array(trace)
    production = trace[int(burn_in * len(trace)):]
    blocks = block_average(production)
    return {
        "start": e_start,
        "mean": production.mean(),
        "std": production.std(),
        "sem": blocks["sem"],
        "sem_naive": blocks["sem_naive"],
        "n_eff": blocks["n_eff"],
        "block_size": blocks["block_size"],
        "n_production": production.size,
        "acc_rate": 100.0 * accepted / n_steps,
        "trace": trace,
        "final_coords": coords,
    }


# ------------------------------------------------------------------------------
# 6. Statistics
# ------------------------------------------------------------------------------
def block_average(samples, min_blocks=4):
    """
    Standard error of the mean of a correlated series by block averaging.

    The series is cut into blocks of 1, 2, 4, ... samples (keeping at least
    `min_blocks` blocks). Block means are less correlated than single samples,
    so their scatter gives an honest SEM once blocks are longer than the
    correlation time. The largest SEM over block sizes is returned as a
    conservative estimate. n_eff = N * (sem_naive / sem)^2 is the number of
    effectively independent samples.
    """
    x = np.asarray(samples, dtype=float)
    n = x.size
    if n < 2:
        return {"mean": float(x.mean()) if n else np.nan, "sem": np.nan,
                "sem_naive": np.nan, "n_eff": float(n), "block_size": 1}
    sem_naive = x.std(ddof=1) / np.sqrt(n)
    best_sem, best_size = sem_naive, 1
    size = 1
    while n // size >= min_blocks:
        n_blocks = n // size
        means = x[: n_blocks * size].reshape(n_blocks, size).mean(axis=1)
        sem = means.std(ddof=1) / np.sqrt(n_blocks)
        if sem > best_sem:
            best_sem, best_size = sem, size
        size *= 2
    n_eff = n * (sem_naive / best_sem) ** 2 if best_sem > 0 else float(n)
    return {"mean": float(x.mean()), "sem": float(best_sem), "sem_naive": float(sem_naive),
            "n_eff": float(n_eff), "block_size": best_size}


def summarize_seeds(results):
    """Mean and sample standard deviation across independent runs (one per seed)."""
    means = np.array([r["mean"] for r in results])
    acc = np.array([r["acc_rate"] for r in results])
    spread = means.std(ddof=1) if means.size > 1 else np.nan
    acc_spread = acc.std(ddof=1) if acc.size > 1 else np.nan
    return {"n_seeds": means.size, "mean": means.mean(), "std": spread,
            "sem": spread / np.sqrt(means.size), "acc_mean": acc.mean(), "acc_std": acc_spread}


# ------------------------------------------------------------------------------
# 7. Main
# ------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdb", nargs="?", default="data/raw/1DUZ.pdb")
    ap.add_argument("--mode", choices=["qm", "standard"], default="qm")
    ap.add_argument("--add-hydrogens", action="store_true", help="protonate with PDBFixer (needed for qm)")
    ap.add_argument("--basis", default="3-21g")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--temp", type=float, default=300.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-seeds", type=int, default=1,
                    help="independent runs with seeds seed, seed+1, ...; reports mean and spread")
    args = ap.parse_args()
    if args.n_seeds < 1:
        ap.error("--n-seeds must be >= 1")

    print(f"=== Q-pMHC toy model [mode: {args.mode.upper()}] ===")

    pdb = add_hydrogens(args.pdb) if args.add_hydrogens else args.pdb
    keep_h = args.add_hydrogens or args.mode == "qm"
    pep, rec = load_complex(pdb, keep_h=keep_h)
    print(f"[1] Loaded: {len(pep['xyz'])} peptide atoms, {len(rec['xyz'])} pocket atoms "
          f"(within {POCKET_CUTOFF} A).")

    rec_q = assign_toy_charges(rec)
    classical_q = assign_toy_charges(pep)

    if args.mode == "qm":
        print(f"[2] HF/{args.basis} on the isolated peptide (gas phase, no embedding)...")
        pep_q, net = compute_qm_charges(pep, basis=args.basis)
        print(f"    Formal net charge: {net:+d}, sum of Mulliken charges: {pep_q.sum():+.3f}")
        mask = classical_q != 0.0
        if mask.sum() > 2:
            r = np.corrcoef(pep_q[mask], classical_q[mask])[0, 1]
            print(f"    Correlation QM vs toy-classical charges (atoms in table): r = {r:.2f}")
    else:
        print("[2] Using toy classical charges.")
        pep_q = classical_q
        print(f"    Sum of peptide charges: {pep_q.sum():+.3f} (toy table, not neutral by construction)")
    print(f"    First 5 peptide charges: {np.round(pep_q[:5], 4)}")

    e0, ec, el = interaction_energy(pep["xyz"], pep_q, rec["xyz"], rec_q)
    print(f"[3] Starting structure: E = {e0:.1f} kcal/mol (Coulomb {ec:.1f}, LJ {el:.1f})")

    print(f"[4] Metropolis MC: {args.steps} steps at {args.temp:.0f} K...")
    seeds = [args.seed + i for i in range(args.n_seeds)]
    runs = [run_monte_carlo(pep["xyz"], pep_q, rec["xyz"], rec_q, n_steps=args.steps,
                            temp=args.temp, rng=np.random.default_rng(s)) for s in seeds]
    res = runs[0]

    print("\n--- SUMMARY ---")
    if args.n_seeds == 1:
        print(f"Mean interaction energy (after burn-in): {res['mean']:.2f} +/- {res['std']:.2f} kcal/mol (std of samples)")
        print(f"Standard error of the mean (block averaging): {res['sem']:.2f} kcal/mol "
              f"[{res['n_production']} samples, ~{res['n_eff']:.0f} effectively independent]")
        print(f"Acceptance rate: {res['acc_rate']:.1f} %")
    else:
        for s, r in zip(seeds, runs):
            print(f"  seed {s}: mean {r['mean']:.2f} +/- {r['sem']:.2f} (SEM, block), "
                  f"acceptance {r['acc_rate']:.1f} %")
        agg = summarize_seeds(runs)
        print(f"Across {agg['n_seeds']} seeds: mean {agg['mean']:.2f} kcal/mol, "
              f"spread (std) {agg['std']:.2f}, SEM {agg['sem']:.2f}")
        print(f"Acceptance rate across seeds: {agg['acc_mean']:.1f} +/- {agg['acc_std']:.1f} %")
    print("Reminder: toy interaction energy, not a binding free energy.")


if __name__ == "__main__":
    main()