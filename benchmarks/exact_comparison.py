"""
exact_comparison.py — Project Aether Benchmark
===============================================
Compares the Project Aether VMC engine against a reference Exact
Diagonalization (ED) solver for the Heisenberg XXX model on a 1-D chain.

Physics notation (enforced by copilot-instructions.md):
  H      — Hamiltonian operator
  psi    — variational ansatz (RBM / NQS)
  sigma  — spin configuration
  theta  — variational parameters
  J      — Heisenberg coupling constant (J > 0 → antiferromagnetic)
  N      — number of lattice sites
  E0     — exact ground-state energy from ED
  E_var  — variational energy from VMC

Usage
-----
    python benchmarks/exact_comparison.py

The script prints a comparison table and exits with a non-zero code if
|E_var - E0| > TOLERANCE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import netket as nk
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Heisenberg XXX parameters
J: float = 1.0          # antiferromagnetic coupling (dimensionless)
N: int = 10              # number of sites on the chain
N_SAMPLES: int = 1_000   # MC samples per VMC step
N_ITER: int = 300        # VMC optimisation steps

# Convergence tolerance: |E_var - E0| must be below this threshold
TOLERANCE: float = 1e-3

GOLDEN_FILE = Path(__file__).parent / "energy_convergence_data.json"


# ---------------------------------------------------------------------------
# Step 1 — Build the Hamiltonian (supported interaction only)
# ---------------------------------------------------------------------------

def build_H_heisenberg(N: int, J: float) -> tuple:
    """Return (H, hi) for the Heisenberg XXX model on a 1-D chain.

    Parameters
    ----------
    N : int
        Number of lattice sites.
    J : float
        Exchange coupling. J > 0 is antiferromagnetic.

    Returns
    -------
    H : nk.operator.Heisenberg
        The Hamiltonian in the NetKet operator format.
    hi : nk.hilbert.Spin
        The Hilbert space.
    """
    graph = nk.graph.Chain(length=N, pbc=True)
    hi = nk.hilbert.Spin(s=0.5, N=graph.n_nodes)
    H = nk.operator.Heisenberg(hilbert=hi, graph=graph, J=J)
    return H, hi


# ---------------------------------------------------------------------------
# Step 2 — Exact Diagonalization reference solver
# ---------------------------------------------------------------------------

def solve_ED(H) -> float:
    """Return the ground-state energy E0 via exact diagonalization.

    Parameters
    ----------
    H : nk.operator.AbstractOperator
        The Hamiltonian whose lowest eigenvalue is sought.

    Returns
    -------
    E0 : float
        Ground-state energy per site (dimensionless).
    """
    H_dense = H.to_dense()
    eigenvalues = np.linalg.eigvalsh(H_dense)
    E0 = float(eigenvalues[0]) / N
    return E0


# ---------------------------------------------------------------------------
# Step 3 — VMC with RBM ansatz
# ---------------------------------------------------------------------------

def run_VMC(H, hi, N_samples: int = N_SAMPLES, n_iter: int = N_ITER) -> float:
    """Run VMC optimisation and return the converged variational energy per site.

    Parameters
    ----------
    H : nk.operator.AbstractOperator
        The Hamiltonian to optimise against.
    hi : nk.hilbert.AbstractHilbert
        The Hilbert space.
    N_samples : int
        Number of Markov-Chain Monte Carlo samples per step.
    n_iter : int
        Number of gradient descent iterations.

    Returns
    -------
    E_var : float
        Converged variational energy per site (dimensionless).
    """
    # Neural-network ansatz: RBM with alpha=1
    psi = nk.models.RBM(alpha=1, param_dtype=complex)

    # Metropolis sampler over spin flips
    sampler = nk.sampler.MetropolisLocal(hi, n_chains=16)

    # Variational state
    vstate = nk.vqs.MCState(sampler, psi, n_samples=N_samples)

    # Stochastic Reconfiguration optimiser
    optimizer = nk.optimizer.Sgd(learning_rate=0.01)
    sr = nk.optimizer.SR(diag_shift=0.1)

    # VMC driver
    gs = nk.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=sr)
    gs.run(n_iter=n_iter, out=None)

    E_var = float(vstate.expect(H).mean.real) / N
    return E_var


# ---------------------------------------------------------------------------
# Step 4 — Golden-file comparison
# ---------------------------------------------------------------------------

def load_golden(path: Path) -> dict:
    """Load the golden-file baseline results."""
    with path.open() as fh:
        return json.load(fh)


def compare_against_golden(E_var: float, golden: dict) -> None:
    """Print a comparison between the current run and the golden-file baseline."""
    baseline_E_var = golden["heisenberg_xxx"]["E_var_per_site"]
    baseline_E0 = golden["heisenberg_xxx"]["E0_per_site"]
    delta_golden = abs(E_var - baseline_E_var)
    print("\n--- Golden-file comparison ---")
    print(f"  Golden E_var / site : {baseline_E_var:.6f}")
    print(f"  This run E_var/site : {E_var:.6f}")
    print(f"  Golden E0    / site : {baseline_E0:.6f}")
    print(f"  |ΔE_var|            : {delta_golden:.2e}")
    if delta_golden > TOLERANCE:
        print(f"  WARNING: deviation {delta_golden:.2e} exceeds tolerance {TOLERANCE:.2e}")
    else:
        print("  PASS: within golden-file tolerance.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    print(f"Project Aether — ED vs VMC Benchmark")
    print(f"  Model  : Heisenberg XXX, 1-D chain, PBC")
    print(f"  N      : {N} sites")
    print(f"  J      : {J}")
    print(f"  Target : |E_var - E0| <= {TOLERANCE}\n")

    H, hi = build_H_heisenberg(N, J)

    print("Running Exact Diagonalization …")
    E0 = solve_ED(H)
    print(f"  E0 / site (ED)  = {E0:.8f}")

    print("\nRunning VMC optimisation …")
    E_var = run_VMC(H, hi)
    print(f"  E_var / site    = {E_var:.8f}")

    delta = abs(E_var - E0)
    print(f"\n  |E_var - E0|    = {delta:.2e}")

    passed = delta <= TOLERANCE
    if passed:
        print("  RESULT: PASS ✓")
    else:
        print(f"  RESULT: FAIL ✗  (tolerance = {TOLERANCE:.2e})")

    if GOLDEN_FILE.exists():
        golden = load_golden(GOLDEN_FILE)
        compare_against_golden(E_var, golden)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
