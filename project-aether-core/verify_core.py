import aether_engine.core as ae
import asyncio
from pydantic import BaseModel

async def verify():
    print("🚀 Initializing Lab-Grade Physics Audit...")
    
    # Create a mock request object
    class MockData:
        atom_count = 10
        interaction_strength = 2.0
    
    # Run the simulation directly from your installed core
    print("🔬 Running Variational Monte Carlo (N=10, J=2.0)...")
    result = await ae.simulate_material(MockData())
    
    print("\n✅ SIMULATION COMPLETE")
    print(f"📊 Ground State Energy: {result['energy']:.6f}")
    print(f"📉 Variance: {result['variance']:.6E}")
    print(f"🛡️ System Status: {result['status']}")
    
    if result['variance'] < 0.1:
        print("\n🌟 VERIFICATION PASSED: The Wavefunction has converged.")
    else:
        print("\n⚠️ WARNING: High variance detected. Check optimizer settings.")

if __name__ == "__main__":
    asyncio.run(verify())
