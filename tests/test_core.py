import numpy as np

from qpmhc import core


def test_coulomb_energy_calculation():
    _, coulomb, _ = core.interaction_energy(
        pep_xyz=np.array([[0.0, 0.0, 0.0]]),
        pep_q=np.array([1.0]),
        rec_xyz=np.array([[2.0, 0.0, 0.0]]),
        rec_q=np.array([-1.0]),
    )

    expected = -core.COULOMB_CONST / (core.EPS_SLOPE * 2.0**2)
    assert np.isclose(coulomb, expected)


def test_metropolis_acceptance(monkeypatch):
    trial_energies = iter([1.0, 0.0])

    def controlled_energy(*_args):
        return next(trial_energies), 0.0, 0.0

    monkeypatch.setattr(core, "interaction_energy", controlled_energy)
    result = core.run_monte_carlo(
        pep_xyz=np.array([[0.0, 0.0, 0.0]]),
        pep_q=np.array([1.0]),
        rec_xyz=np.array([[2.0, 0.0, 0.0]]),
        rec_q=np.array([-1.0]),
        n_steps=1,
        temp=300.0,
        rng=np.random.default_rng(17),
    )

    assert result["acc_rate"] == 100.0
    assert np.array_equal(result["trace"], np.array([1.0, 0.0]))


def test_charge_array_shape():
    atoms = {
        "resname": ["ALA", "ALA", "LYS", "ASP"],
        "name": ["N", "CA", "NZ", "OD2"],
    }

    charges = core.assign_toy_charges(atoms)

    assert charges.shape == (len(atoms["name"]),)
    assert np.isclose(charges[2], 1.0)
    assert np.isclose(charges[3], -0.5)
