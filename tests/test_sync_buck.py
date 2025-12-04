import networkx as nx
from env.circuit_env import CircuitEnv
from env.components import Inductor, Capacitor, VoltageSource, Switch, Diode
from utils.mode_analysis import analyze_switching_modes

def test_sync_buck():
    print("Building Synchronous Buck Converter...")
    
    # 1. Define Inventory
    # Sync Buck uses 2 Switches (S1, S2) instead of 1 Switch + 1 Diode
    inventory = [
        VoltageSource(name="Vin", nodes=(0,0), dc_value=12.0),  # 0
        Switch(name="S1", nodes=(0,0)),                         # 1: High Side
        Switch(name="S2", nodes=(0,0)),                         # 2: Low Side
        Inductor(name="L1", nodes=(0,0), value=47e-6),          # 3
        VoltageSource(name="Vout", nodes=(0,0), dc_value=5.0)   # 4
    ]
    
    env = CircuitEnv(initial_components=inventory)
    env.reset()
    
    # 2. Construct Topology
    # Nodes: 0(GND), 1(Vin+), 2(SW), 3(Vout+)
    
    # Note: reset() already adds intermediate nodes 1, 2, 3...
        
    # Place Components
    print("Placing Vin (1 -> 0)...")
    # Action: [1, 0, 1, 0]
    env.step([1, 0, 1, 0])
    
    print("Placing S1 (High Side, 1 -> 2)...")
    # Action: [1, 1, 1, 2]
    env.step([1, 1, 1, 2])
    
    print("Placing S2 (Low Side, 2 -> 0)...")
    # Note: S2 replaces the Diode. Connected between SW(2) and GND(0).
    # Action: [1, 2, 2, 0]
    env.step([1, 2, 2, 0])
    
    print("Placing Inductor (2 -> 3)...")
    # Action: [1, 3, 2, 3]
    env.step([1, 3, 2, 3])
    
    print("Placing Vout (3 -> 0)...")
    # Action: [1, 4, 3, 0]
    env.step([1, 4, 3, 0])
    
    print("Circuit Constructed.")
    
    # 3. Analyze Modes
    print("\nAnalyzing Switching Modes (S1, S2)...")
    modes = analyze_switching_modes(env.circuit_graph)
    
    for state, data in modes.items():
        switches = data['switches']
        valid = data['valid']
        reasons = data['reasons']
        trends = data['inductor_trends']
        
        # Format state string
        state_str = ", ".join([f"{name}={'ON' if s else 'OFF'}" for name, s in zip(switches, state)])
        
        print(f"\nMode: [{state_str}]")
        print(f"  Valid: {valid}")
        if not valid:
            print(f"  Issues: {reasons}")
        print(f"  Inductor Trends: {trends}")

if __name__ == "__main__":
    test_sync_buck()
