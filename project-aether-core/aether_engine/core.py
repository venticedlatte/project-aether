"""
run_phase12_styled.py — The Aether Cloud (CORS Unlocked)
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import netket as nk
import numpy as np

app = FastAPI(title="Project Aether Engine", version="Alpha 1.0")

# ─── CRITICAL FIX: ENABLE CORS ────────────────────────────────────────────────
# Allows React frontend (localhost:5173) to talk to this Python API
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

@app.get("/")
def health():
    return {"status": "Aether Engine Online", "version": "Alpha 1.0"}

@app.post("/simulate_material")
async def simulate_material(data: SimulateRequest):
    L = data.atom_count
    g = nk.graph.Chain(length=L, pbc=False)
    hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)
    H = nk.operator.Ising(hilbert=hi, graph=g, h=1.0, J=data.interaction_strength)

    ma = nk.models.RBM(alpha=1, param_dtype=float)
    sa = nk.sampler.MetropolisLocal(hi)
    op = nk.optimizer.Sgd(learning_rate=0.05)
    # FIX: optimizer belongs in VMC driver only, not MCState
    vstate = nk.vqs.MCState(sa, ma, n_samples=500)

    vmc = nk.driver.VMC(H, op, variational_state=vstate)
    vmc.run(n_iter=50)

    return {
        "energy": float(vstate.expect(H).mean.real),
        "variance": float(vstate.expect(H).variance.real),
        "status": "Stable"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
