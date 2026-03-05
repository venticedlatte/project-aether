import aether_engine.core as ae
import asyncio

async def verify():
    print("🚀 Initializing Alpha 1.1 Physics Audit...")
    
    # We now use a proper SimulateRequest to match the core.py logic
    class MockRequest:
        atom_count = 10
        interaction_strength = 2.0
        precision = "lab"  # This triggers 2048 samples / 150 iterations
    
    print(f"🔬 Running Lab-Grade VMC (Samples: 2048, Iterations: 150)...")
    result = await ae.simulate_material(MockRequest())
    
    print("\n✅ AUDIT COMPLETE")
    print(f"📊 Ground State Energy: {result['energy']:.6f}")
    print(f"📉 Variance: {result['variance']:.6E}")
    print(f"🛡️ System Status: {result['status']}")
    print(f"⚙️ Mode: {result['precision_mode']}")

if __name__ == "__main__":
    asyncio.run(verify())
