import networkx as nx
from env.circuit_env import CircuitEnv
from env.components import Inductor, Capacitor, VoltageSource, Switch, Diode
from utils.mode_analysis import analyze_switching_modes

def test_open_inductor():
    print("Building Open Inductor Circuit (Vin -> S1 -> L1 -> S2 -> GND)...")
    
    # Inventory: Vin, S1, S2, L1
    inventory = [
        VoltageSource(name="Vin", nodes=(0,0), dc_value=12.0),
        Switch(name="S1", nodes=(0,0)),
        Switch(name="S2", nodes=(0,0)),
        Inductor(name="L1", nodes=(0,0), value=47e-6)
    ]
    
    env = CircuitEnv(initial_components=inventory)
    env.reset()
    
    # Construct Topology
    # Nodes: 0(GND), 1(Vin), 2(L_in), 3(L_out)
    
    # 1. Vin (1 -> 0)
    print("Placing Vin (1 -> 0)...")
    env.step([1, 0, 1, 0])
    
    # 2. S1 (1 -> 2)
    # n1=1, n2=2. Body Diode: 2 -> 1. Blocks 1->2.
    print("Placing S1 (1 -> 2)...")
    env.step([1, 1, 1, 2])
    
    # 3. L1 (2 -> 3)
    print("Placing L1 (2 -> 3)...")
    env.step([1, 3, 2, 3])
    
    # 4. S2 (3 -> 0) - REMOVED to create Open Circuit
    # n1=3, n2=0. Body Diode: 0 -> 3. Blocks 3->0.
    # print("Placing S2 (3 -> 0)...")
    # env.step([1, 2, 3, 0])
    
    print("Circuit Constructed.")
    
    # Analyze Modes
    print("\nAnalyzing Switching Modes...")
    modes = analyze_switching_modes(env.circuit_graph)
    
    for state, data in modes.items():
        switches = data['switches']
        valid = data['valid']
        reasons = data['reasons']
        trends = data['inductor_trends']
        
        state_str = ", ".join([f"{name}={'ON' if s else 'OFF'}" for name, s in zip(switches, state)])
        
        print(f"\nMode: [{state_str}]")
        print(f"  Valid: {valid}")
        if not valid:
            print(f"  Issues: {reasons}")
        print(f"  Inductor Trends: {trends}")

if __name__ == "__main__":
    test_open_inductor()
