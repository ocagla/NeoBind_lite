"""Generate a standard-mode Monte Carlo trace figure from a PDB input."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from qpmhc import core


def main() -> int:
    """Run standard-mode sampling and write the energy-trace figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=core.DEFAULT_SEED)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "results" / "standard_mc_trace.svg",
    )
    args = parser.parse_args()

    peptide, receptor = core.load_complex(args.pdb)
    peptide_charges = core.assign_toy_charges(peptide)
    receptor_charges = core.assign_toy_charges(receptor)
    result = core.run_monte_carlo(
        peptide["xyz"],
        peptide_charges,
        receptor["xyz"],
        receptor_charges,
        n_steps=args.steps,
        temp=core.DEFAULT_TEMPERATURE,
        rng=np.random.default_rng(args.seed),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(np.arange(result["trace"].size), result["trace"], linewidth=1.2)
    axis.set_xlabel("Monte Carlo step")
    axis.set_ylabel("Interaction energy (kcal/mol)")
    axis.set_title("Standard-charge rigid-body sampling")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output, format="svg")
    plt.close(figure)

    print(f"Wrote {args.output}")
    print(
        f"Mean after burn-in: {result['mean']:.2f} +/- "
        f"{result['std']:.2f} kcal/mol (std of samples); "
        f"SEM (block averaging): {result['sem']:.2f} kcal/mol; "
        f"acceptance: {result['acc_rate']:.1f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
