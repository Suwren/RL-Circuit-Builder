import numpy as np
from env.circuit_env import CircuitEnv
from env.components import Inductor, Capacitor, VoltageSource, Switch, Diode
from utils.loop_analysis import find_loops_and_direction

def test_buck_loops():
    print("Building Buck Converter for Loop Analysis...")
    
    # 1. Define Inventory (Same as test_buck.py)
    inventory = [
        VoltageSource(name="Vin", nodes=(0,0), dc_value=12.0),  # 0
        Switch(name="S1", nodes=(0,0), state=True),             # 1
        Diode(name="D1", nodes=(0,0)),                          # 2
        Inductor(name="L1", nodes=(0,0), value=47e-6),          # 3
        VoltageSource(name="Vout", nodes=(0,0), dc_value=5.0)   # 4
    ]
    
    env = CircuitEnv(initial_components=inventory)
    env.reset()
    
    # 2. Construct Topology (Same as test_buck.py)
    # Nodes: 0(GND), 1(Vin+), 2(SW), 3(Vout+)
    
    # Add Nodes
    for _ in range(3):
        env.step({"type": 0, "component_idx": 0, "nodes": [0, 0]})
        
    # Place Components
    env.step({"type": 1, "component_idx": 0, "nodes": [1, 0]}) # Vin (1->0)
    env.step({"type": 1, "component_idx": 1, "nodes": [1, 2]}) # S1 (1->2)
    env.step({"type": 1, "component_idx": 2, "nodes": [0, 2]}) # D1 (0->2) (Anode 0, Cathode 2)
    env.step({"type": 1, "component_idx": 3, "nodes": [2, 3]}) # L1 (2->3)
    env.step({"type": 1, "component_idx": 4, "nodes": [3, 0]}) # Vout (3->0)
    
    print("Circuit Constructed.")
    print("Edges:", env.circuit_graph.edges(data=True))
    
    # 3. Analyze Loops
    print("\nAnalyzing Loops...")
    loops = find_loops_and_direction(env.circuit_graph)
    
    for i, loop in enumerate(loops):
        print(f"\nLoop {i+1}:")
        print(f"  Path (Nodes): {loop['nodes']}")
        print(f"  Components: {loop['components']}")
        print(f"  Net Voltage Source EMF: {loop['net_voltage']} V")
        print(f"  Direction Interpretation: Current flows along {loop['nodes']}")

if __name__ == "__main__":
    test_buck_loops()
