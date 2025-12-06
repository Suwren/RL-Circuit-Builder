import torch
import numpy as np
import os
import sys
import networkx as nx

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from env.circuit_env import CircuitEnv
from env.components import VoltageSource, Switch, Inductor, Capacitor, Resistor, Diode
from alphazero.model import AlphaZeroNet
from alphazero.mcts import MCTS
from alphazero.circuit_wrapper import CircuitEnvWrapper

def evaluate_trained_model(model_path="models/alphazero_iter_41.pth"):
    """
    Loads a trained AlphaZero model, generates a circuit, and prints detailed evaluation.
    """
    print(f"Loading model from {model_path}...")
    
    # 1. Setup Environment
    max_nodes = 12 # Match training config (12*12=144 features for value head)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define Inventory (Must match training configuration in trainer.py)
    # Based on create_buck_circuit.py and trainer.py context
    # Define Inventory (Must match training configuration in trainer.py)
    # Updated for Multi-Port Expansion
    inventory = []
    # 3 Voltage Sources
    inventory.append(VoltageSource(name="V1", nodes=(0, 0), value=20.0, dc_value=20.0))
    inventory.append(VoltageSource(name="V2", nodes=(0, 0), value=10.0, dc_value=10.0))
    inventory.append(VoltageSource(name="V3", nodes=(0, 0), value=5.0, dc_value=5.0))
    # 1 Inductor
    inventory.append(Inductor(name="L1", nodes=(0, 0), value=47e-6))
    # 4 Switches
    for i in range(4):
        inventory.append(Switch(name=f"S{i+1}", nodes=(0, 0)))
        
    # Initialize Environment with verbose=True for detailed logging
    raw_env = CircuitEnv(initial_components=inventory, max_nodes=max_nodes, verbose=True)
    env = CircuitEnvWrapper(raw_env)
    
    # 2. Load Model
    # Input channels: 9 (Adjacency) + 3 (Inventory Counts) + 4 (Node Features) = 16
    input_channels = 16 
    model = AlphaZeroNet(input_shape=(input_channels, max_nodes, max_nodes), 
                              num_actions=env.action_space_size()).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    else:
        print(f"Error: Model {model_path} not found!")
        return
        
    model.eval()
    
    # 3. MCTS Generation
    # Use more simulations for inference
    mcts = MCTS(model, num_simulations=200, device=device)
    
    obs, info = env.reset()
    done = False
    step = 0
    
    print("\n=== Start Circuit Generation ===")
    
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
        
        type_idx = action[1]
        comp_name = "Unknown"
        if type_idx == 0: comp_name = "VoltageSource"
        elif type_idx == 1: comp_name = "Inductor"
        elif type_idx == 2: comp_name = "Switch"
            
        if action[0] == 0:
            print(f"Step {step}: Action=STOP")
        else:
            print(f"Step {step}: Action=Place {comp_name} at Node {action[2]} -> Node {action[3]}")
        
        # Execute step
        # Note: env.step_flat returns 0 reward usually, we check terminal state
        env.step_flat(action_idx)
        
        terminated, score = env.is_terminal()
        if terminated:
            done = True
            print(f"\n=== Generation Complete ===")
            
            # The environment's _calculate_circuit_score will print detailed logs because verbose=True
            # We don't need to call it manually again if is_terminal called it, 
            # BUT is_terminal calls it internally and returns normalized score.
            # However, CircuitEnvWrapper.is_terminal calls env._calculate_circuit_score().
            # Since verbose=True, it should have already printed the logs during the last step/check.
            # Let's double check if we need to call it explicitly to be sure we see the final result clearly.
            
            print("\n--- Final Circuit Evaluation ---")
            real_score = raw_env._calculate_circuit_score()
            print(f"Final Raw Score: {real_score}")
            
            print("\nCircuit Edges:")
            for u, v, data in raw_env.circuit_graph.edges(data=True):
                comp = data.get('component')
                comp_name = comp.name if comp else "Unknown"
                details = ""
                if isinstance(comp, Switch):
                    n1, n2 = comp.nodes
                    details = f"[Drain: {n1}, Source: {n2}, Body Diode: {n2}->{n1}]"
                elif isinstance(comp, VoltageSource):
                    pos, neg = comp.nodes
                    details = f"[Pos: {pos}, Neg: {neg}]"
                elif isinstance(comp, Inductor):
                    n1, n2 = comp.nodes
                    details = f"[{n1}-{n2}]"
                    
                print(f"  {u} <-> {v} : {comp_name} {details}")

if __name__ == "__main__":
    evaluate_trained_model()
