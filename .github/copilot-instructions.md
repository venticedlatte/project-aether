# GitHub Copilot Instructions — Project Aether

## 1. Project Context

Project Aether is an **aerospace-grade** JAX-accelerated Neural Quantum State (NQS) engine for many-body quantum simulation. It is **not** a generic machine-learning library. The primary KPI is energy accuracy: the absolute variational energy error $|\Delta E| \approx 10^{-27}$ J in physical units. Every code suggestion must be evaluated against this precision requirement first.

The primary KPI is variational energy accuracy: the error $|\Delta E|$ between the VMC estimate and the Exact Diagonalization (ED) reference must remain below $10^{-27}$ J in absolute physical units (or $10^{-4}$ in dimensionless units for benchmark systems).

The engine targets the variational solution of quantum Hamiltonians (Heisenberg, Ising, and extensions) using Variational Monte Carlo (VMC) with neural-network ansätze, built on top of **NetKet** and **JAX**.

---

## 2. JAX-Native Performance Patterns

All numerically intensive code **must** use JAX primitives. Never fall back to plain NumPy loops or Python-level iteration over array elements.

### Required patterns

| Pattern | Requirement |
|---|---|
| `jax.jit` | Wrap every top-level function that will be called repeatedly (e.g., energy estimators, gradient steps). |
| `jax.vmap` | Vectorise over batches of samples or spin configurations — never use explicit `for` loops over samples. |
| `jax.lax.scan` | Use for sequential Monte Carlo sweeps instead of Python `for` loops. |
| `jax.grad` / `jax.value_and_grad` | Compute gradients of the variational energy. Do not implement finite-difference gradients. |

### Example skeleton

```python
import jax
import jax.numpy as jnp

@jax.jit
def local_energy(params, sigma: jnp.ndarray) -> jnp.ndarray:
    """Return the local energy E_loc(sigma) for a single sample sigma."""
    ...

# Vectorise over a batch of samples with vmap
batch_local_energy = jax.vmap(local_energy, in_axes=(None, 0))
```

---

## 3. Physics-First Coding Style

### 3.1 Variable naming — standard physics notation (mandatory)

Use the symbols physicists recognise. Non-standard names will be rejected in review.

| Concept | Required name | Forbidden alternatives |
|---|---|---|
| Hamiltonian operator / matrix | `H` | `hamiltonian`, `ham`, `operator` |
| Wavefunction / ansatz amplitude | `psi` or `Psi` | `wavefunction`, `wave_fn`, `amplitude` |
| Spin configuration / basis state | `sigma` | `state`, `config`, `sample_vec` |
| Variational parameters | `theta` | `params`, `weights`, `w` |
| Coupling constant (Heisenberg) | `J` | `coupling`, `j_val` |
| External magnetic field | `h` | `field`, `B_ext` |
| Ground-state energy | `E0` | `ground_energy`, `e_gs` |
| Variational energy estimate | `E_var` | `energy`, `loss` |
| Number of sites | `N` | `n_sites`, `num_sites` |
| Hilbert space | `hi` | `hilbert`, `space` |

### 3.2 Function and module naming

- Functions that build or return a Hamiltonian are named `build_H_*` (e.g., `build_H_heisenberg`).
- Functions that compute energies are named `compute_E_*`.
- Modules are snake_case but retain physics acronyms: `vmc_engine.py`, `nqs_ansatz.py`, `ed_solver.py`.

### 3.3 Units and constants

- Energy is always in **Joules** unless an explicit `_dimensionless` suffix is used.
- Use `scipy.constants` or define a single `constants.py` file. Never hard-code physical constants inline.

---

## 4. Anti-Hallucination Rules — Supported Interactions Only

This engine supports a **fixed, vetted set** of Hamiltonians. Do **not** generate code for an unsupported interaction; instead, emit a `NotImplementedError` with a reference to the relevant literature or NetKet documentation.

### 4.1 Supported Hamiltonians

| Model | NetKet constructor | Notes |
|---|---|---|
| Heisenberg XXX (isotropic) | `nk.operator.Heisenberg` | `J` parameter; periodic or open boundary |
| Heisenberg XXZ (anisotropic) | `nk.operator.Heisenberg` with `anisotropy` kwarg | `J`, `Delta` parameters |
| Transverse-Field Ising | `nk.operator.IsingJax` | `J`, `h` parameters |

### 4.2 Unsupported interaction protocol

If a contributor or agent requests an interaction **not** in the table above, generate the following stub and do not attempt to implement the physics:

```python
def build_H_<model_name>(*args, **kwargs):
    raise NotImplementedError(
        "<ModelName> is not yet validated in Project Aether. "
        "Open a Physics Enhancement Request and provide: "
        "(1) the Hamiltonian definition, "
        "(2) a peer-reviewed reference, "
        "(3) expected ground-state energy for a benchmark system. "
        "See CONTRIBUTING.md for the full checklist."
    )
```

### 4.3 Geometry support

Only the following lattice geometries are supported:
- 1-D chain (`nk.graph.Chain`)
- 2-D square lattice (`nk.graph.Square`)
- 3-D cubic lattice (`nk.graph.Hypercube` with `n_dim=3`)

For any other geometry, raise `NotImplementedError` with the same protocol as §4.2.

---

## 5. Accuracy and Convergence Standards

- The target variational energy error versus Exact Diagonalization (ED) is $|E_\text{var} - E_0| \leq 10^{-4}$ in dimensionless units for benchmark systems ($N \leq 20$ sites).
- Every new model or ansatz **must** include a convergence test in `benchmarks/exact_comparison.py`.
- The golden-file baseline is stored in `benchmarks/energy_convergence_data.json`. Any PR that degrades the golden-file energy by more than $10^{-4}$ will be blocked by the `physics-audit` CI workflow.

---

## 6. Code Quality Checklist for Every PR

Before submitting, verify:

- [ ] All new functions are decorated with `@jax.jit` where appropriate.
- [ ] No Python `for` loops over sample batches — use `jax.vmap` or `jax.lax.scan`.
- [ ] All variable names comply with §3.1.
- [ ] No new Hamiltonians are introduced without a corresponding ED benchmark.
- [ ] The Hamiltonian under test passes the Hermitian check: `‖H − H†‖ < 1e-10`.
- [ ] `benchmarks/energy_convergence_data.json` is updated if a new baseline is established.
- [ ] No physical constants are hard-coded; use `constants.py`.
