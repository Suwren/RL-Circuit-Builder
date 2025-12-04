import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.circuit_env import CircuitEnv
from env.components import VoltageSource, Switch, Inductor

def test_blank_canvas_logic():
    print("Testing Blank Canvas Logic...")
    
    # Setup inventory
    inventory = [
        VoltageSource(name="Vin", nodes=(0,0), dc_value=12.0),
        Switch(name="S1", nodes=(0,0)),
        Switch(name="S2", nodes=(0,0)),
        Inductor(name="L1", nodes=(0,0), value=47e-6),
        VoltageSource(name="Vout", nodes=(0,0), dc_value=5.0)
    ]
    
    env = CircuitEnv(initial_components=inventory, max_nodes=10)
    obs, _ = env.reset()
    
    print(f"Initial State: Nodes={env.circuit_graph.number_of_nodes()}, Edges={env.circuit_graph.number_of_edges()}")
    assert env.circuit_graph.number_of_nodes() == 0
    
    # --- Step 1: Place First Component (Vin) ---
    # Action: [1, 0, 0, 0] (Nodes ignored for first component)
    action = [1, 0, 0, 0]
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\nStep 1 (First Comp): Nodes={env.circuit_graph.number_of_nodes()}, Edges={env.circuit_graph.number_of_edges()}")
    assert env.circuit_graph.number_of_nodes() == 2 # Should be Node 0 and 1
    assert env.circuit_graph.has_edge(0, 1)
    assert env.node_counter == 2
    
    # --- Step 2: Place Second Component (S1) - Connect to Existing + New ---
    # Current Nodes: 0, 1. Next New Node ID: 2.
    # Action: Connect Node 1 (Existing) to Node 2 (New)
    # Action Vector: [1, 1, 1, 2]
    action = [1, 1, 1, 2]
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\nStep 2 (Connect Existing-New): Nodes={env.circuit_graph.number_of_nodes()}, Edges={env.circuit_graph.number_of_edges()}")
    assert env.circuit_graph.number_of_nodes() == 3 # Should be Node 0, 1, 2
    assert env.circuit_graph.has_edge(1, 2)
    assert env.node_counter == 3
    
    # --- Step 3: Place Third Component (S2) - Connect to Existing + Existing ---
    # Current Nodes: 0, 1, 2.
    # Action: Connect Node 2 (Existing) to Node 0 (Existing) -> Form Loop
    # Action Vector: [1, 2, 2, 0]
    action = [1, 2, 2, 0]
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\nStep 3 (Connect Existing-Existing): Nodes={env.circuit_graph.number_of_nodes()}, Edges={env.circuit_graph.number_of_edges()}")
    assert env.circuit_graph.number_of_nodes() == 3 # No new nodes
    assert env.circuit_graph.has_edge(2, 0)
    
    # --- Step 4: Invalid Action - Connect New + New ---
    # Current Nodes: 0, 1, 2. Next New Node ID: 3.
    # Action: Connect Node 3 (New) to Node 4 (New)
    # Action Vector: [1, 3, 3, 4]
    action = [1, 3, 3, 4]
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\nStep 4 (Invalid New-New): Reward={reward}")
    assert reward < 0 # Should be penalized
    # Graph should not change
    assert env.circuit_graph.number_of_edges() == 3
    
    print("\nTest Passed!")

if __name__ == "__main__":
    test_blank_canvas_logic()
