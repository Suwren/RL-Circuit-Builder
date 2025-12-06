import networkx as nx
from env.circuit_env import CircuitEnv
from env.components import VoltageSource, Switch, Diode, Inductor, Capacitor, Resistor

def create_buck_circuit():
    # 1. Define Component Inventory
    # We need: Vin, Switch, Diode, Inductor, Capacitor, Resistor
    inventory = [
        VoltageSource(name="V1", nodes=(1, 0), dc_value=10.0, role="input"),   # Index 0: Pos=1, Neg=0
        Switch(name="S1", nodes=(0, 0)),                                       # Index 1
        Switch(name="S2", nodes=(0, 0)),                                       # Index 2 (Replaces Diode)
        Inductor(name="L1", nodes=(0, 0), value=100e-6),                       # Index 3
        VoltageSource(name="V2", nodes=(3, 0), dc_value=5.0, role="output"),   # Index 4: Pos=3, Neg=0
        VoltageSource(name="V3", nodes=(0, 0), dc_value=5.0, role="output")    # Index 5: Dummy V3
    ]

    # 2. Initialize Environment
    # max_nodes needs to be enough for 0, 1, 2, 3 (so at least 4)
    env = CircuitEnv(initial_components=inventory, max_nodes=10, verbose=True)
    
    print("Initialized CircuitEnv with inventory:")
    for i, comp in enumerate(inventory):
        print(f"  {i}: {comp.name} ({type(comp).__name__})")

    # 3. Manually build the Buck Converter
    # Topology:
    # V1: 0 (GND) -> 1 (Input)
    # S1:  1 -> 2
    # S2:  0 -> 2 (Synchronous Rectifier)
    # L1:  2 -> 3
    # V2: 0 -> 3 (Load)
    # V3: Just placed floating or connected for inventory check
    
    # Add nodes
    env.circuit_graph.add_node(0, type="Ground") # 0 is usually treated as reference/ground implicitly or explicitly
    env.circuit_graph.add_node(1, type="Intermediate")
    env.circuit_graph.add_node(2, type="Intermediate")
    env.circuit_graph.add_node(3, type="Intermediate")
    env.node_counter = 4
    
    # Add components
    # V1 (Index 0) -> Nodes 1, 0 (Pos, Neg)
    env._add_component_to_graph(0, 1, 0)
    
    # S1 (Index 1) -> Nodes 1, 2
    env._add_component_to_graph(1, 1, 2)
    
    # S2 (Index 2) -> Nodes 2, 0
    env._add_component_to_graph(2, 2, 0)
    
    # L1 (Index 3) -> Nodes 2, 3
    env._add_component_to_graph(3, 2, 3)
    
    # V2 (Index 4) -> Nodes 3, 0 (Pos, Neg)
    env._add_component_to_graph(4, 3, 0)
    
    # V3 (Index 5) -> Nodes 0, 0 (Short/Floating - just present for logic)
    env._add_component_to_graph(5, 0, 0)
    
    print("\nCircuit Constructed. Nodes:", env.circuit_graph.nodes())
    print("Edges:")
    for u, v, data in env.circuit_graph.edges(data=True):
        comp = data['component']
        details = ""
        if isinstance(comp, VoltageSource):
            # Nodes: (Pos, Neg)
            details = f" [Pos: {comp.nodes[0]}, Neg: {comp.nodes[1]}]"
        elif isinstance(comp, Switch):
            # Nodes: (Drain, Source) - Assumption
            # Body Diode: Source -> Drain (N-Channel MOSFET convention usually)
            # If current flows S->D when OFF (body diode), then Anode=Source, Cathode=Drain.
            details = f" [Drain: {comp.nodes[0]}, Source: {comp.nodes[1]}, Body Diode: {comp.nodes[1]}->{comp.nodes[0]}]"
        elif isinstance(comp, Diode):
            # Nodes: (Anode, Cathode)
            details = f" [Anode: {comp.nodes[0]}, Cathode: {comp.nodes[1]}]"
            
        print(f"  {u} <-> {v} : {comp.name} ({type(comp).__name__}){details}")

    # 4. Verify and Score
    print("\nCalculating Score...")
    try:
        score = env._calculate_circuit_score()
        print(f"Final Score: {score}")
    except Exception as e:
        print(f"Error calculating score: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    create_buck_circuit()
