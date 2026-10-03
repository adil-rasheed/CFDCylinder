"""Sanity checks for the LBM solver. Run: python test_lbm.py"""
import numpy as np

from cylinder_lbm import C, OPP, W, equilibrium, macroscopic, run


def test_lattice():
    assert np.isclose(W.sum(), 1.0)
    assert np.all(C[OPP] == -C)


def test_equilibrium_moments():
    rho = np.full((4, 3), 1.1)
    u = np.zeros((2, 4, 3))
    u[0], u[1] = 0.05, -0.02
    r, v = macroscopic(equilibrium(rho, u))
    assert np.allclose(r, rho) and np.allclose(v, u)


def test_uniform_flow_without_cylinder():
    r = run(steps=500, nx=120, ny=40, cylinder=False, probe=(60, 20), kick=0.0, log_every=0, verbose=False)
    assert np.isfinite(r["u"]).all()
    assert np.abs(r["u"][0] - r["U"]).max() < 1e-4
    assert np.abs(r["u"][1]).max() < 1e-4
    assert r["mass_drift"] < 1e-6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
