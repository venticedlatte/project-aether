---
name: "Physics Bug Report"
about: "Report a discrepancy between Aether's result and expected physics."
title: "[Physics Bug] <short description>"
labels: ["physics-bug", "needs-triage"]
assignees: []
---

<!--
  Please answer every section below. Vague "it doesn't work" reports will be
  closed without investigation. Precise physics context is required so that
  the team can reproduce and diagnose the issue.
-->

## 1. Simulation Parameters

| Parameter | Value |
|---|---|
| **Model** | (e.g., Heisenberg XXX, Heisenberg XXZ, Transverse-Field Ising) |
| **Geometry** | (Chain / Square / Cubic — with size, e.g., `Chain(N=16, pbc=True)`) |
| **Coupling constant J** | |
| **Anisotropy Δ** (XXZ only) | |
| **External field h** | |
| **Ansatz** | (e.g., RBM alpha=1, GCNN) |
| **Number of VMC samples** | |
| **Number of optimisation steps** | |
| **JAX version** | |
| **NetKet version** | |

---

## 2. Expected Result

_What should Aether produce according to theory?_

<!-- If a closed-form or Bethe-ansatz result is known, state it here.
     If you ran an independent ED calculation, paste the result.
     If the expected result is unknown, state that explicitly. -->

**Theoretical / ED ground-state energy (E0 per site):**

```
E0 / site = ______ (source: ______)
```

---

## 3. Observed Result

_What did Aether actually produce?_

**Converged variational energy (E_var per site):**

```
E_var / site = ______
```

**Absolute deviation |E_var − E0|:**

```
|E_var - E0| = ______
```

Paste the full convergence log or the last 10 lines of stdout below:

```
<paste output here>
```

---

## 4. Reproducibility

- [ ] I ran `python benchmarks/exact_comparison.py` and it produces the same discrepancy.
- [ ] I can reproduce this on a clean environment.
- [ ] I have verified that my Hamiltonian is Hermitian (‖H − H†‖ < 1e-10).

**Minimal reproduction script:**

```python
# paste the smallest possible script that triggers the bug
```

---

## 5. Additional Context

<!-- Any other relevant information: hardware, OS, stack trace, etc. -->
