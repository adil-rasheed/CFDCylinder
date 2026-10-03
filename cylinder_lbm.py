"""2D flow past a cylinder with a D2Q9 Lattice Boltzmann (BGK) solver.

Records the cross-stream velocity at a wake probe so the shedding frequency
(Strouhal number) can be extracted by analyze.py.

Boundary conditions:
  inlet  (x=0)    : prescribed velocity, Zou-He style reconstruction
  outlet (x=nx-1) : zero-gradient for the unknown (left-moving) populations
  top/bottom      : periodic (from np.roll streaming)
  cylinder        : full-way bounce-back
"""
import argparse
import os
import time

import numpy as np

# D2Q9 lattice: velocities, weights and opposite directions
C = np.array([(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1),
              (1, 1), (-1, 1), (-1, -1), (1, -1)])
W = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
RIGHT = np.where(C[:, 0] == 1)[0]   # populations entering from the inlet
LEFT = np.where(C[:, 0] == -1)[0]   # populations entering from the outlet
MID = np.where(C[:, 0] == 0)[0]


def equilibrium(rho, u):
    """f_eq_i = w_i rho (1 + 3 c.u + 4.5 (c.u)^2 - 1.5 |u|^2)."""
    cu = 3.0 * np.einsum("id,dxy->ixy", C, u)
    usq = 1.5 * (u[0] ** 2 + u[1] ** 2)
    return rho[None] * W[:, None, None] * (1.0 + cu + 0.5 * cu ** 2 - usq[None])


def macroscopic(f):
    rho = f.sum(axis=0)
    u = np.einsum("id,ixy->dxy", C, f) / rho[None]
    return rho, u


def run(re=100.0, steps=30000, nx=420, ny=160, D=20, U=0.04,
        cylinder=True, probe=None, kick=0.3, log_every=1000, verbose=True):
    """Run the simulation. Returns a dict with probe history and final fields."""
    nu = U * D / re
    tau = 3.0 * nu + 0.5
    omega = 1.0 / tau

    # Cylinder slightly off the centerline to trigger the instability
    cx, cy = 4 * D, ny // 2 + 1
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    solid = ((x - cx) ** 2 + (y - cy) ** 2 < (D / 2) ** 2) if cylinder \
        else np.zeros((nx, ny), dtype=bool)

    if probe is None:
        probe = (cx + 2 * D, ny // 2)
    px, py = probe

    # Inlet profile with a tiny perturbation
    vel = np.zeros((2, nx, ny))
    vel[0] = U * (1.0 + 1e-4 * np.sin(2 * np.pi * y / (ny - 1)))

    # Initial state: inlet profile plus a transverse kick in the near wake,
    # which breaks the symmetry and shortens the onset of shedding
    u0 = vel.copy()
    u0[1] = kick * U * np.exp(-((x - cx - D) ** 2 + (y - cy) ** 2) / D ** 2)
    u0[:, solid] = 0.0
    fin = equilibrium(np.ones((nx, ny)), u0)

    probe_v = np.empty(steps)
    mass0 = fin.sum()
    if verbose:
        print(f"Re={re:g}  grid={nx}x{ny}  D={D}  U={U}  nu={nu:.5f}  tau={tau:.4f}")
    t0 = time.time()

    for it in range(steps):
        # Outlet: zero-gradient for the unknown left-moving populations
        fin[LEFT, -1, :] = fin[LEFT, -2, :]

        rho, u = macroscopic(fin)

        # Inlet: impose velocity, density from the known populations
        u[:, 0, :] = vel[:, 0, :]
        rho[0, :] = (fin[MID, 0, :].sum(axis=0) + 2.0 * fin[LEFT, 0, :].sum(axis=0)) \
            / (1.0 - u[0, 0, :])

        feq = equilibrium(rho, u)
        # Non-equilibrium bounce-back for the unknown inlet populations
        fin[RIGHT, 0, :] = feq[RIGHT, 0, :] + fin[OPP[RIGHT], 0, :] - feq[OPP[RIGHT], 0, :]

        probe_v[it] = u[1, px, py]

        # BGK collision
        fout = fin - omega * (fin - feq)

        # Bounce-back on the cylinder
        for i in range(9):
            fout[i, solid] = fin[OPP[i], solid]

        # Streaming (periodic in y via np.roll)
        for i in range(9):
            fin[i] = np.roll(np.roll(fout[i], C[i, 0], axis=0), C[i, 1], axis=1)

        if log_every and (it + 1) % log_every == 0:
            umax = np.sqrt(u[0] ** 2 + u[1] ** 2).max()
            if not np.isfinite(umax):
                raise FloatingPointError(f"Simulation diverged at step {it + 1}")
            if verbose:
                print(f"  step {it + 1:6d}/{steps}  |u|max={umax:.4f}  "
                      f"v_probe={probe_v[it]:+.5f}  ({time.time() - t0:.0f}s)")

    rho, u = macroscopic(fin)
    return dict(probe_v=probe_v, u=u, rho=rho, solid=solid, re=re, D=D, U=U,
                nu=nu, tau=tau, probe=probe, mass_drift=abs(fin.sum() - mass0) / mass0,
                runtime=time.time() - t0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--re", type=float, default=100.0)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--nx", type=int, default=420)
    p.add_argument("--ny", type=int, default=160)
    p.add_argument("--D", type=int, default=20)
    p.add_argument("--U", type=float, default=0.04)
    p.add_argument("--kick", type=float, default=0.3,
                   help="initial transverse wake perturbation, as a fraction of U")
    p.add_argument("--out", default="results")
    a = p.parse_args()

    r = run(a.re, a.steps, a.nx, a.ny, a.D, a.U, kick=a.kick)
    os.makedirs(a.out, exist_ok=True)
    tag = f"re{a.re:g}"
    np.savez_compressed(
        os.path.join(a.out, f"run_{tag}.npz"),
        probe_v=r["probe_v"].astype(np.float32),
        u=r["u"].astype(np.float32), solid=r["solid"],
        re=r["re"], D=r["D"], U=r["U"], probe=np.array(r["probe"]))
    print(f"Done in {r['runtime']:.0f}s -> {a.out}/run_{tag}.npz")


if __name__ == "__main__":
    main()
