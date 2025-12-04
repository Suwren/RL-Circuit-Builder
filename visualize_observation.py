import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.circuit_env import CircuitEnv
from env.components import VoltageSource, Switch, Inductor, Capacitor, Resistor

def get_component_name_by_id(type_id):
    mapping = {
        0: "None",
        1: "Wire",
        2: "Resistor",
        3: "Inductor",
        4: "Capacitor",
        5: "V_Source",
        6: "I_Source",
        7: "Switch",
        8: "Diode"
    }
    return mapping.get(type_id, str(type_id))

def visualize_agent_observation():
    print("Initializing Environment...")
    
    # 1. Setup Environment
    inventory = [
        VoltageSource(name="Vin", nodes=(0,0), dc_value=12.0), # ID 5
        Switch(name="S1", nodes=(0,0)),                        # ID 7
        Inductor(name="L1", nodes=(0,0), value=47e-6),         # ID 3
        VoltageSource(name="Vout", nodes=(0,0), dc_value=5.0)  # ID 5
    ]
    
    env = CircuitEnv(initial_components=inventory, max_nodes=6)
    obs, _ = env.reset()
    
    # 2. Perform some actions to build a structure
    # Step 1: Place Vin (creates Node 0, 1)
    # Action: [1, 0, 0, 0]
    print("Step 1: Placing Vin...")
    obs, _, _, _, _ = env.step([1, 0, 0, 0])
    
    # Step 2: Place S1 connecting Node 1 to New Node (Node 2)
    # Action: [1, 1, 1, 2]
    print("Step 2: Placing S1 (Node 1 -> Node 2)...")
    obs, _, _, _, _ = env.step([1, 1, 1, 2])
    
    # Step 3: Place L1 connecting Node 2 to Node 0 (Loop)
    # Action: [1, 2, 2, 0]
    print("Step 3: Placing L1 (Node 2 -> Node 0)...")
    obs, _, _, _, _ = env.step([1, 2, 2, 0])
    
    # 3. Extract Observation Data
    adj_matrix = obs['adjacency']
    node_feats = obs['node_features']
    inv_mask = obs['inventory_mask']
    
    print("\n--- Agent Observation Analysis ---")
    print(f"Adjacency Matrix Shape: {adj_matrix.shape}")
    print(f"Node Features Shape: {node_feats.shape}")
    print(f"Inventory Mask: {inv_mask}")
    
    # 4. Visualize Adjacency Matrix
    plt.figure(figsize=(10, 8))
    
    # Create annotations (Component Names)
    annot = np.empty(adj_matrix.shape, dtype=object)
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            val = adj_matrix[i, j]
            annot[i, j] = get_component_name_by_id(val) if val > 0 else ""
            
    sns.heatmap(adj_matrix, annot=annot, fmt="", cmap="Blues", cbar=False, linewidths=.5, square=True)
    plt.title("Agent Observation: Adjacency Matrix (Component Types)")
    plt.xlabel("Node ID")
    plt.ylabel("Node ID")
    
    output_file = "agent_observation_vis.png"
    plt.savefig(output_file)
    print(f"\nVisualization saved to: {output_file}")
    
    # 5. Print Node Features
    print("\nNode Features (Degree, Connected):")
    for i in range(len(node_feats)):
        # Only print for active nodes (degree > 0) or first few
        if node_feats[i][0] > 0:
            print(f"Node {i}: Degree={node_feats[i][0]}, Connected={node_feats[i][1]}")

if __name__ == "__main__":
    visualize_agent_observation()
