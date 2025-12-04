import networkx as nx
from env.circuit_env import CircuitEnv
from env.components import Inductor, Capacitor, VoltageSource, Switch, Diode
from utils.mode_analysis import analyze_switching_modes

def test_modes():
    print("Building Circuit with 2 Switches for Mode Analysis...")
    
    # Define Inventory: Vin, S1, S2, L1, R_Load (Vout)
    # A hypothetical circuit: Vin -> S1 -> L1 -> S2 -> GND
    inventory = [
        VoltageSource(name="Vin", nodes=(0,0), dc_value=10.0),
        Switch(name="S1", nodes=(0,0)),
        Switch(name="S2", nodes=(0,0)),
        Inductor(name="L1", nodes=(0,0), value=10e-6),
        VoltageSource(name="Vout", nodes=(0,0), dc_value=5.0)
    ]
    
    env = CircuitEnv(initial_components=inventory)
    env.reset()
    
    # Construct Topology
    # Nodes: 0, 1, 2, 3
    for _ in range(3): env.step({"type": 0, "component_idx": 0, "nodes": [0,0]})
    
    # Vin: 1 -> 0
    env.step({"type": 1, "component_idx": 0, "nodes": [1, 0]})
    # S1: 1 -> 2
    env.step({"type": 1, "component_idx": 1, "nodes": [1, 2]})
    # L1: 2 -> 3
    env.step({"type": 1, "component_idx": 3, "nodes": [2, 3]})
    # S2: 3 -> 0
    env.step({"type": 1, "component_idx": 2, "nodes": [3, 0]})
    
    # Add an illegal short for testing: Switch S3 across Vin (1 -> 0)
    # We need to add S3 to inventory first? 
    # The inventory in test_modes.py only has S1, S2.
    # Let's just short Vin with S1 by placing S1 at [1, 0] instead?
    # No, let's stick to the current valid circuit first to confirm the "Increasing" fix.
    # To test illegal loop, we can modify the graph manually or add a new test case.
    # Let's add a separate test function for illegal loop.
    
    print("Circuit Constructed.")
    print("Edges:", env.circuit_graph.edges(data=True))
    
    # Analyze Modes
    print("\nAnalyzing Switching Modes...")
    modes = analyze_switching_modes(env.circuit_graph)
    
    print(f"Found {len(modes)} modes.")
    
    for state, data in modes.items():
        switches = data['switches']
        graph = data['graph']
        valid = data['valid']
        reasons = data['reasons']
        trends = data['inductor_trends']
        
        # Format state string
        state_str = ", ".join([f"{name}={'ON' if s else 'OFF'}" for name, s in zip(switches, state)])
        
        print(f"\nMode: [{state_str}]")
        print(f"  Valid: {valid}")
        if reasons:
            print(f"  Issues: {reasons}")
        print(f"  Inductor Trends: {trends}")

def test_illegal_loop():
    print("\nTesting Illegal Loop Detection...")
    # Construct a circuit with Vin shorted by a switch
    inventory = [
        VoltageSource(name="Vin", nodes=(0,0), dc_value=10.0),
        Switch(name="S1", nodes=(0,0))
    ]
    env = CircuitEnv(initial_components=inventory)
    env.reset()
    
    # Nodes: 0, 1
    env.step({"type": 0, "component_idx": 0, "nodes": [0,0]})
    
    # Vin: 1 -> 0
    env.step({"type": 1, "component_idx": 0, "nodes": [1, 0]})
    # S1: 1 -> 0 (Shorting Vin)
    env.step({"type": 1, "component_idx": 1, "nodes": [1, 0]})
    
    print("Circuit Constructed (Vin || S1).")
    
    modes = analyze_switching_modes(env.circuit_graph)
    
    # Mode [S1=ON] should be invalid
    for state, data in modes.items():
        switches = data['switches']
        state_str = ", ".join([f"{name}={'ON' if s else 'OFF'}" for name, s in zip(switches, state)])
        print(f"Mode: [{state_str}] -> Valid: {data['valid']}")
        if not data['valid']:
            print(f"  Reason: {data['reasons']}")

if __name__ == "__main__":
    test_modes()
    test_illegal_loop()
