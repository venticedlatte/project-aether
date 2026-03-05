""" project_aether_core/aether_engine/core.py — Precision Upgrade """
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import netket as nk
import numpy as np

app = FastAPI(title="Project Aether Engine", version="Alpha 1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SimulateRequest(BaseModel):
    atom_count: int
    interaction_strength: float
    precision: str = "standard"  # Options: 'standard' or 'lab'


@app.get("/")
def health():
    return {"status": "Aether Engine Online", "version": "Alpha 1.1"}


@app.post("/simulate_material")
async def simulate_material(data: SimulateRequest):
    L = data.atom_count

    # --- PRECISION CONFIGURATION ---
    # 'lab' mode increases samples and iterations for scientific validity
    if data.precision == "lab":
        n_samples = 2048
        n_iter = 150
        lr = 0.01
    else:
        n_samples = 512
        n_iter = 50
        lr = 0.05

    g = nk.graph.Chain(length=L, pbc=False)
    hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)
    H = nk.operator.Ising(hilbert=hi, graph=g, h=1.0, J=data.interaction_strength)

    ma = nk.models.RBM(alpha=1, param_dtype=float)
    sa = nk.sampler.MetropolisLocal(hi)
    op = nk.optimizer.Sgd(learning_rate=lr)

    vstate = nk.vqs.MCState(sa, ma, n_samples=n_samples)
    vmc = nk.driver.VMC(H, op, variational_state=vstate)

    # Execute simulation
    vmc.run(n_iter=n_iter)

    energy = float(vstate.expect(H).mean.real)
    variance = float(vstate.expect(H).variance.real)

    return {
        "energy": energy,
        "variance": variance,
        "status": "Converged" if variance < 0.01 else "Estimated",
        "precision_mode": data.precision
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
