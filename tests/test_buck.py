import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from env.circuit_env import CircuitEnv
from env.components import Inductor, Capacitor, VoltageSource, Switch, Diode, Resistor

def test_buck_construction():
    print("Building Buck Converter...")
    
    # 1. Define Inventory for Buck Converter
    # We need: Vin, Switch, Diode, Inductor, Load Voltage Source (Vout)
    # Capacitor removed per user request
    inventory = [
        VoltageSource(name="Vin", nodes=(0,0), dc_value=12.0),  # 0: Input 12V
        Switch(name="S1", nodes=(0,0), state=True),             # 1: Switch (Closed for test)
        Diode(name="D1", nodes=(0,0)),                          # 2: Freewheeling Diode
        Inductor(name="L1", nodes=(0,0), value=47e-6),          # 3: Inductor 47uH
        VoltageSource(name="Vout", nodes=(0,0), dc_value=5.0)   # 4: Load Voltage Source (Target Output)
    ]
    
    env = CircuitEnv(initial_components=inventory)
    obs, _ = env.reset()
    
    # 2. Construct Topology
    # Target Nodes:
    # 0: GND
    # 1: Vin+
    # 2: SW Node (between S, D, L)
    # 3: Vout Node (between L, Vout)
    
    # Step 1: Add necessary nodes (We start with 0. Need 1, 2, 3)
    print("Adding Nodes...")
    for _ in range(3):
        env.step({"type": 0, "component_idx": 0, "nodes": [0, 0]})
    
    print(f"Current Nodes: {env.circuit_graph.nodes()}")
    
    # Step 2: Place Components
    # Vin: Positive at 1, Negative at 0
    print("Placing Vin (1 -> 0)...")
    env.step({"type": 1, "component_idx": 0, "nodes": [1, 0]})
    
    # Switch: Between Vin+ (1) and SW (2)
    print("Placing Switch (1 -> 2)...")
    env.step({"type": 1, "component_idx": 1, "nodes": [1, 2]})
    
    # Diode: Anode at GND (0), Cathode at SW (2)
    print("Placing Diode (0 -> 2)...")
    env.step({"type": 1, "component_idx": 2, "nodes": [0, 2]})
    
    # Inductor: Between SW (2) and Vout (3)
    print("Placing Inductor (2 -> 3)...")
    env.step({"type": 1, "component_idx": 3, "nodes": [2, 3]})
    
    # Load Voltage Source (Vout): Positive at 3, Negative at 0
    print("Placing Load Voltage Source Vout (3 -> 0)...")
    # Note: Component index for Vout is now 4 because Capacitor was removed
    obs, _, _, _, _ = env.step({"type": 1, "component_idx": 4, "nodes": [3, 0]})
    
    print("Buck Converter Constructed!")
    print("Edges:", env.circuit_graph.edges(data=True))
    
    # 3. Visualize Adjacency Matrix
    adj_matrix = obs["adjacency"]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(adj_matrix, cmap='Greys', interpolation='nearest')
    plt.title("Buck Converter Adjacency Matrix")
    plt.xlabel("Node Index")
    plt.ylabel("Node Index")
    plt.colorbar(label="Connection (1=Connected)")
    
    # Add grid lines for clarity
    plt.grid(which='major', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    plt.xticks(np.arange(len(adj_matrix)))
    plt.yticks(np.arange(len(adj_matrix)))
    
    output_file = "buck_adjacency.png"
    plt.savefig(output_file)
    print(f"Adjacency matrix plot saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    try:
        test_buck_construction()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
