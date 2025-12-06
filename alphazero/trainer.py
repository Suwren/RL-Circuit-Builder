import os
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
import random
import time
import warnings

# Suppress DirectML performance warnings
warnings.filterwarnings("ignore", message="The operator 'aten::native_group_norm_backward' is not currently supported")

# Add project root to path to ensure imports work correctly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.circuit_env import CircuitEnv
from env.components import VoltageSource, Switch, Inductor
from alphazero.model import AlphaZeroNet
from alphazero.mcts import MCTS
from alphazero.circuit_wrapper import CircuitEnvWrapper
from utils.device_helper import get_device

class AlphaZeroTrainer:
    """
    Trainer class for the AlphaZero agent.
    Manages the self-play loop, data collection, and neural network training.
    """
    def __init__(self, num_sources=2, num_inductors=1, max_nodes=12):
        # 1. Configuration Parameters
        self.max_nodes = max_nodes
        self.num_simulations = 300  # Balanced search depth (User Request)
        self.batch_size = 64
        self.epochs = 20 # Increased from 10 to learn more from limited data
        self.lr = 1e-3
        self.device = get_device()
        
        # 2. Initialize Environment and Inventory
        # We create a dynamic inventory based on the requested components
        self.inventory = self._create_inventory(num_sources, num_inductors)
        self.raw_env = CircuitEnv(initial_components=self.inventory, max_nodes=self.max_nodes)
        self.env = CircuitEnvWrapper(self.raw_env)
        
        # 3. Initialize Neural Network and MCTS
        # Input channels: 6 (Directional Adjacency) + 3 (Inventory Counts) + 4 (Node Features) = 13
        input_channels = 13
        self.model = AlphaZeroNet(input_shape=(input_channels, self.max_nodes, self.max_nodes), 
                                  num_actions=self.env.action_space_size()).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.mcts = MCTS(self.model, cpuct=1.0, num_simulations=self.num_simulations, device=self.device)
        
        # Replay Buffer to store self-play examples
        # Reduced size to keep data fresh and learn from recent high-reward episodes
        self.replay_buffer = deque(maxlen=1000)
        
        # [NEW] Duplicate Topology Tracking
        self.visited_topologies = set()

    def _create_inventory(self, num_sources=None, num_inductors=None):
        """
        Creates the component inventory.
        Updated for Multi-Port Expansion:
        - 3 Voltage Sources (20V, 10V, 5V)
        - 1 Inductor
        - 6 Switches
        """
        inventory = []
        
        # 1. Voltage Sources
        # V1 = 20V
        inventory.append(VoltageSource(name="V1", nodes=(0, 0), value=20.0, dc_value=20.0))
        # V2 = 10V
        inventory.append(VoltageSource(name="V2", nodes=(0, 0), value=10.0, dc_value=10.0))
        # V3 = 5V
        inventory.append(VoltageSource(name="V3", nodes=(0, 0), value=5.0, dc_value=5.0))
        
        # 2. Inductors
        inventory.append(Inductor(name="L1", nodes=(0, 0), value=47e-6))
        
        # 3. Switches (5 Switches)
        for i in range(5):
            inventory.append(Switch(name=f"S{i+1}", nodes=(0, 0)))
            
        return inventory

    def self_play(self, temp=1.0):
        """
        Executes one episode of self-play.
        Returns a list of training examples: [(observation, policy, value), ...]
        """
        self.mcts.clear() # Reset MCTS tree for the new episode
        self.env = CircuitEnvWrapper(CircuitEnv(self.inventory, self.max_nodes)) # Reset environment
        examples = []
        
        step = 0
        while True:
            step += 1
            
            # Debug: Verify clone consistency to ensure MCTS works correctly
            s_orig = self.env.canonical_string()
            s_clone = self.env.clone().canonical_string()
            if s_orig != s_clone:
                print(f"FATAL: Clone Mismatch at step {step}!")
                print(f"Orig: {s_orig}")
                print(f"Clone: {s_clone}")
                raise RuntimeError("Clone Mismatch")

            # Temperature is passed as argument
            pi = self.mcts.get_action_prob(self.env, temp=temp)
            
            # Store observation and policy. Value (z) will be filled after the episode ends.
            obs_tensor = self.env.get_obs_tensor().numpy()
            examples.append([obs_tensor, pi, None])
            
            # Choose action based on the policy distribution
            if temp == 0:
                action = np.argmax(pi)
            else:
                action = np.random.choice(len(pi), p=pi)
            
            # Execute action in the environment
            reward = self.env.step_flat(action)
            
            # Check for termination
            terminated, final_score = self.env.is_terminal()
            
            if terminated:
                # [NEW] Duplicate Topology Check
                # Calculate topology hash (invariant to Switch IDs)
                topo_hash = self._get_topology_hash(self.env.env.circuit_graph)
                
                if topo_hash in self.visited_topologies:
                    # Penalty for duplicate topology
                    final_score = -200.0
                    print(f"  [Duplicate] Topology {topo_hash[:8]} already seen. Penalizing.")
                else:
                    self.visited_topologies.add(topo_hash)
                
                # Episode finished. Backpropagate the final score as the value for all steps.
                # Note: This treats the problem as a single-player optimization task.
                return [(x[0], x[1], final_score) for x in examples]

    def _get_topology_hash(self, graph):
        """
        Computes a hash for the circuit topology using Weisfeiler-Lehman algorithm.
        Handles MultiGraph by converting edges to nodes in a bipartite-like structure.
        Masks Switch IDs to ensure S1/S2 permutations are treated as identical.
        """
        import networkx as nx
        from env.components import Switch
        
        # Transformation: Convert Edge-Labeled MultiGraph -> Node-Labeled Graph
        # Original Nodes -> Nodes with label "Node"
        # Edges -> Nodes with label "Component_Type" connected to original nodes
        
        G_prime = nx.Graph()
        
        # 1. Add Original Nodes
        for n in graph.nodes():
            G_prime.add_node(f"n_{n}", label="Node")
            
        # 2. Convert Components (Edges) to Nodes
        # Use edge keys to handle parallel edges uniquely
        for u, v, k, d in graph.edges(keys=True, data=True):
            comp = d.get('component')
            if comp:
                if isinstance(comp, Switch):
                    type_label = "Switch"
                else:
                    type_label = comp.name
            else:
                type_label = "Wire"
                
            # Create a unique node for this component
            comp_node_id = f"c_{u}_{v}_{k}"
            G_prime.add_node(comp_node_id, label=type_label)
            
            # Connect component node to circuit nodes
            G_prime.add_edge(f"n_{u}", comp_node_id)
            G_prime.add_edge(f"n_{v}", comp_node_id)
            
        # 3. Compute Hash on the transformed simple graph
        # explicit node_attr='label'
        return nx.weisfeiler_lehman_graph_hash(G_prime, node_attr='label')

    def train(self, num_iterations=10, episodes_per_iter=5):
        """
        Main training loop.
        """
        print(f"Starting AlphaZero training on {self.device}...")
        
        for i in range(num_iterations):
            print(f"\n=== Iteration {i+1}/{num_iterations} ===")
            
            # 1. Self-Play: Collect new data
            print("Self-playing...")
            new_examples = []
            from tqdm import tqdm
            for e in tqdm(range(episodes_per_iter), desc="Episodes"):
                # Dynamic Temperature Schedule based on Episode Index
                if e < 10:
                    current_temp = 2.0 # High exploration (First 10 episodes)
                elif e < 15:
                    current_temp = 1 # Medium exploration (Next 5 episodes)
                else:
                    current_temp = 0.5 # Low exploration (Last 5+ episodes)

                start_time = time.time()
                episode_data = self.self_play(temp=current_temp)
                new_examples.extend(episode_data)
                duration = time.time() - start_time
                
                # Check score of this episode
                # episode_data is list of (obs, pi, z). z is the final score.
                episode_score = episode_data[0][2]
                
                tqdm.write(f"  Episode {e+1}: {len(episode_data)} steps, Reward={episode_score:.4f}, Temp={current_temp}, Time={duration:.1f}s")
                
                # [NEW] Save ANY good circuit found during training
                if episode_score > 0.5:
                    self.save_best_circuit(self.env, episode_score, f"{i+1}_ep_{e+1}")

            self.replay_buffer.extend(new_examples)
            print(f"Buffer size: {len(self.replay_buffer)}")
            
            # 2. Training: Update Neural Network
            print("Training network...")
            self.train_network()
            
            # 3. Save Model Checkpoint
            torch.save(self.model.state_dict(), f"models/alphazero_iter_{i+1}.pth")
            
            # 4. Periodic Evaluation (Deterministic)
            print("Evaluating model (Deterministic)...")
            self.evaluate_model(i+1)

    def evaluate_model(self, iteration):
        """
        Runs a single deterministic episode to evaluate model progress.
        Uses temp=0 to select the best moves.
        """
        self.mcts.clear()
        self.env = CircuitEnvWrapper(CircuitEnv(self.inventory, self.max_nodes))
        
        step = 0
        done = False
        print(f"  [Eval] Start Generation...")
        
        while not done:
            step += 1
            # Use temp=0 for deterministic (greedy) action selection
            pi = self.mcts.get_action_prob(self.env, temp=0)
            action = np.argmax(pi)
            
            # Execute
            self.env.step_flat(action)
            terminated, score = self.env.is_terminal()
            
            if terminated:
                done = True
                print(f"  [Eval] Iter {iteration}: Steps={step}, Final Score={score:.4f}")
                
                # If good result, save plot
                # User Request: Save if Reward > 0.5
                if score > 0.5:
                    self.save_best_circuit(self.env, score, f"{iteration}_eval")


    def save_best_circuit(self, env, score, iteration):
        """Saves the plot of the best circuit found so far."""
        from utils.visualization import plot_circuit
        filename = f"best_circuit_iter_{iteration}.png"
        print(f"  Saving best circuit with score {score:.4f} to {filename}")
        plot_circuit(env.env.circuit_graph, filename=filename)
        
        # [NEW] Also save text description
        self.save_circuit_text(env, score, iteration)

    def save_circuit_text(self, env, score, iteration):
        """Saves the circuit topology map as a text file."""
        import os
        from env.components import Switch, VoltageSource, Inductor
        
        # specific directory for topologies
        os.makedirs("saved_topologies", exist_ok=True)
        filename = f"saved_topologies/topology_score_{score:.4f}_iter_{iteration}.txt"
        
        with open(filename, "w", encoding='utf-8') as f:
            f.write(f"Circuit Topology (Score: {score:.4f}, Iteration: {iteration})\n")
            f.write("="*50 + "\n\n")
            f.write("Circuit Edges:\n")
            
            # Sort edges for consistent output
            edges = list(env.env.circuit_graph.edges(data=True))
            edges.sort(key=lambda x: (x[0], x[1]))
            
            for u, v, data in edges:
                comp = data.get('component')
                comp_name = comp.name if comp else "Unknown"
                details = ""
                if isinstance(comp, Switch):
                    n1, n2 = comp.nodes
                    # Assuming default diode direction for Switch (Source->Drain or Drain->Source?)
                    # In components.py, Switch usually has body diode.
                    # Let's print the actual nodes connected.
                    # And infer diode direction if needed, but printing nodes is safest.
                    # User output example: [Drain: 1, Source: 0, Body Diode: 0->1]
                    # We'll approximate this format.
                    details = f"[Drain: {n1}, Source: {n2}, Body Diode: {n2}->{n1}]"
                elif isinstance(comp, VoltageSource):
                    pos, neg = comp.nodes
                    details = f"[Pos: {pos}, Neg: {neg}]"
                elif isinstance(comp, Inductor):
                    n1, n2 = comp.nodes
                    details = f"[{n1}-{n2}]"
                
                line = f"  {u} <-> {v} : {comp_name} {details}\n"
                f.write(line)
        
        print(f"  Saved topology description to {filename}")

    def train_network(self):
        """
        Trains the neural network using the replay buffer.
        Minimizes the combined loss: (z - v)^2 - pi^T * log(p) + c||theta||^2
        """
        if len(self.replay_buffer) < self.batch_size:
            return
            
        examples = list(self.replay_buffer)
        random.shuffle(examples)
        
        # Prepare batches
        obs_batch = torch.FloatTensor(np.array([x[0] for x in examples])).to(self.device)
        pi_batch = torch.FloatTensor(np.array([x[1] for x in examples])).to(self.device)
        v_batch = torch.FloatTensor(np.array([x[2] for x in examples])).to(self.device)
        
        dataset = TensorDataset(obs_batch, pi_batch, v_batch)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        total_loss = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            for obs, target_pi, target_v in dataloader:
                pred_pi_logits, pred_v = self.model(obs)
                
                # Value Loss (MSE): (z - v)^2
                loss_v = F.mse_loss(pred_v.view(-1), target_v)
                
                # Policy Loss (Cross Entropy): -pi * log(p)
                log_probs = F.log_softmax(pred_pi_logits, dim=1)
                loss_pi = -torch.sum(target_pi * log_probs) / target_pi.size(0)
                
                loss = loss_v + loss_pi
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            total_loss += epoch_loss
            
        print(f"  Avg Loss: {total_loss / self.epochs:.4f}")

if __name__ == "__main__":
    # Hyperparameters: 50 iterations, 20 episodes per iteration = 1000 total episodes.
    trainer = AlphaZeroTrainer()
    trainer.train(num_iterations=50, episodes_per_iter=20)
