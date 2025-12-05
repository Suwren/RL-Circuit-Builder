import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from env.circuit_env import CircuitEnv
from env.components import VoltageSource, Switch, Inductor
from alphazero.model import AlphaZeroNet
from alphazero.mcts import MCTS
from alphazero.circuit_wrapper import CircuitEnvWrapper
from utils.visualization import plot_circuit

def test_agent(model_path="models/alphazero_iter_6.pth"):
    """
    Loads a trained AlphaZero model and generates a circuit using MCTS.
    """
    # 1. Setup Environment
    max_nodes = 12
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define Inventory (Must match training configuration)
    inventory = []
    inventory.append(VoltageSource("V1", (0,0), dc_value=12.0, role="input"))
    inventory.append(VoltageSource("V2", (0,0), dc_value=5.0, role="output"))
    inventory.append(Inductor("L1", (0,0), value=47e-6))
    for i in range(2):
        inventory.append(Switch(f"S{i+1}", (0,0)))
        
    raw_env = CircuitEnv(initial_components=inventory, max_nodes=max_nodes)
    env = CircuitEnvWrapper(raw_env)
    
    # 2. Load Model
    input_channels = 9 + len(inventory) + 4
    model = AlphaZeroNet(input_shape=(input_channels, max_nodes, max_nodes), 
                              num_actions=env.action_space_size()).to(device)
    
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Model {model_path} not found! Using random weights.")
        
    model.eval()
    
    # 3. MCTS Generation
    # Use more simulations for inference to ensure high quality
    mcts = MCTS(model, num_simulations=100, device=device)
    
    obs, info = env.reset()
    done = False
    step = 0
    
    print("\n=== Start Generation ===")
    
    while not done:
        step += 1
        # Use temp=0 for deterministic (greedy) action selection
        pi = mcts.get_action_prob(env, temp=0)
        action_idx = np.argmax(pi)
        
        # Decode action for display
        action = []
        rem = action_idx
        for dim in reversed(env.dims):
            action.append(rem % dim)
            rem //= dim
        action = list(reversed(action))
        
        comp_idx = action[1]
        comp_name = "Unknown"
        if comp_idx < len(inventory):
            comp_name = inventory[comp_idx].name
            
        if action[0] == 0:
            print(f"Step {step}: Action={action} (STOP)")
        else:
            print(f"Step {step}: Action={action} (Place {comp_name} at {action[2]}-{action[3]})")
        
        # Execute step
        env.step_flat(action_idx)
        
        terminated, score = env.is_terminal()
        if terminated:
            done = True
            print(f"\nGeneration Complete!")
            print(f"Final Normalized Score: {score:.4f}")
            
            # Get real un-normalized score
            if hasattr(raw_env, '_calculate_circuit_score'):
                real_score = raw_env._calculate_circuit_score()
                print(f"Real Circuit Score: {real_score:.4f}")
                
            print("\nFinal Circuit Edges:")
            for u, v, data in raw_env.circuit_graph.edges(data=True):
                comp = data.get('component')
                comp_name = comp.name if comp else "Unknown"
                print(f"  {u} -- {v} : {comp_name}")
            
    # 4. Visualize Result
    print("Plotting circuit...")
    plot_circuit(raw_env.circuit_graph, filename="final_circuit.png")
    print("Saved to final_circuit.png")

if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")
        
    test_agent()
