import networkx as nx
from env.components import VoltageSource, Switch, Inductor
from utils.mode_analysis import check_static_safety

def test_static_leakage():
    print("Testing Static Leakage Detection...")
    
    # Circuit Topology from User:
    # Nodes: 0, 1, 2, 3
    # 0 <-> 1 : S1
    # 0 <-> 2 : L1
    # 0 <-> 3 : S2
    # 1 <-> 3 : V1 (Input)
    # 2 <-> 3 : V2 (Output/Load)
    
    # Let's reconstruct this graph manually
    graph = nx.MultiGraph()
    
    # Components
    v1 = VoltageSource("V1", (1, 3), dc_value=10.0, role="input") # Pos=1, Neg=3 (based on user desc: V1 is 1-3)
    # Wait, user said "1 <-> 3 : V1". Let's assume standard V1 is input.
    # User log says: "输入电源: V1".
    
    s1 = Switch("S1", (0, 1)) # Body diode: 0->1? Or 1->0?
    # Switch default: n1(Drain), n2(Source). Body diode n2->n1.
    # If S1 is (0, 1), body diode is 1->0.
    
    l1 = Inductor("L1", (0, 2), value=100e-6)
    
    s2 = Switch("S2", (0, 3))
    
    v2 = VoltageSource("V2", (2, 3), dc_value=5.0, role="output")
    
    # Add edges
    graph.add_edge(1, 3, component=v1)
    graph.add_edge(0, 1, component=s1)
    graph.add_edge(0, 2, component=l1)
    graph.add_edge(0, 3, component=s2)
    graph.add_edge(2, 3, component=v2)
    
    print("Graph constructed.")
    
    # Check Forward Direction (Source=V1, Load=V2)
    # Path suspected: V1+ (1) -> S1(Body 1->0) -> L1(0->2) -> V2(2->3) -> V1- (3)
    # Note: S1 body diode is Source->Drain. If S1 nodes are (0,1), and 0 is Drain, 1 is Source, then 1->0.
    # Let's verify Switch node assignment in env.
    
    safe, msg = check_static_safety(graph, source_name="V1", load_name="V2")
    print(f"Result: Safe={safe}")
    print(f"Message: {msg}")
    
    if safe:
        print("FAILURE: Should have detected leakage path V1 -> S1 -> L1 -> V2")
    else:
        print("SUCCESS: Leakage detected.")

if __name__ == "__main__":
    test_static_leakage()
