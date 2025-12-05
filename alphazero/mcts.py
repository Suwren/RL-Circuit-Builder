import math
import numpy as np
import torch
from utils.device_helper import get_device

class MCTS:
    """
    Monte Carlo Tree Search (MCTS) engine.
    Uses a neural network to guide the search (AlphaZero style).
    """
    def __init__(self, model, cpuct=1.0, num_simulations=800, device="auto"):
        """
        Args:
            model: Neural network for policy and value prediction.
            cpuct: Exploration constant for PUCT algorithm.
            num_simulations: Number of simulations per search.
            device: Device to run the model on.
        """
        self.model = model
        self.cpuct = cpuct
        self.num_simulations = num_simulations
        self.device = get_device(device)
        
        # Tree Statistics
        # Key: Canonical state string
        self.Qsa = {}       # Q(s,a): Action Value
        self.Nsa = {}       # N(s,a): Action Visit Count
        self.Ns = {}        # N(s): State Visit Count
        self.Ps = {}        # P(s): Prior Policy Probabilities (from NN)
        self.Es = {}        # E(s): Terminal State Status (Reward or None)
        self.Vs = {}        # V(s): Valid Action Mask

    def clear(self):
        """Clears the search tree."""
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}

    def get_action_prob(self, env, temp=1):
        """
        Runs MCTS simulations and returns the action probability distribution.
        
        Args:
            env: The current environment state.
            temp: Temperature parameter (1 for exploration, 0 for exploitation).
        """
        canonical_state = env.canonical_string()
        
        # Ensure root node is expanded
        self.search(env.clone())
        
        # Add Dirichlet noise to the root node for exploration (only during training)
        if temp == 1:
            s = canonical_state
            if s in self.Ps:
                alpha = 0.3 # Dirichlet parameter
                epsilon = 0.25 # Noise weight
                
                valids = self.Vs[s]
                noise = np.random.dirichlet([alpha] * len(self.Ps[s]))
                
                # Mix noise into the prior probabilities
                self.Ps[s] = (1 - epsilon) * self.Ps[s] + epsilon * noise
                self.Ps[s] = self.Ps[s] * valids # Re-apply mask
                self.Ps[s] /= np.sum(self.Ps[s]) # Re-normalize

        # Run simulations
        from tqdm import tqdm
        for _ in tqdm(range(self.num_simulations), desc="MCTS", leave=False):
            env_copy = env.clone() # Clone environment for simulation
            self.search(env_copy)
            
        s = canonical_state
        # Calculate visit counts
        counts = [self.Nsa.get((s, a), 0) for a in range(env.action_space_size())]
        
        # Greedy selection (temp -> 0)
        if temp == 0:
            best_a = np.argmax(counts)
            probs = [0] * len(counts)
            probs[best_a] = 1
            return probs
        
        # Stochastic selection based on visit counts
        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        
        # Handle edge case where search failed (should be rare)
        if counts_sum == 0:
            print("Warning: counts_sum is 0, clearing tree and retrying search...")
            self.clear()
            self.search(env.clone())
            for _ in range(self.num_simulations):
                self.search(env.clone())
                
            counts = [self.Nsa.get((s, a), 0) for a in range(env.action_space_size())]
            counts = [x ** (1. / temp) for x in counts]
            counts_sum = float(sum(counts))
            
            if counts_sum == 0:
                # Fallback to uniform distribution
                probs = [1.0 / len(counts)] * len(counts)
                return probs

        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, env):
        """
        Performs one MCTS simulation (Selection -> Expansion -> Simulation -> Backpropagation).
        Note: Since we use AlphaZero, Simulation is replaced by NN Value estimation.
        """
        s = env.canonical_string()

        # 1. Check for Terminal State
        if s not in self.Es:
            terminated, reward = env.is_terminal()
            self.Es[s] = reward if terminated else None

        if self.Es[s] is not None:
            return self.Es[s]

        # 2. Expansion (Leaf Node)
        if s not in self.Ps:
            # Predict policy and value using Neural Network
            obs_tensor = env.get_obs_tensor().to(self.device).unsqueeze(0)
            
            self.model.eval()
            with torch.no_grad():
                policy_logits, v = self.model(obs_tensor)
            
            policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            v = v.item()
            
            # Mask invalid actions
            valids = env.get_action_mask()
            policy_probs = policy_probs * valids
            sum_probs = np.sum(policy_probs)
            
            if sum_probs > 0:
                policy_probs /= sum_probs
            else:
                print("Warning: All valid moves were masked, doing workaround.")
                policy_probs = valids / np.sum(valids)

            self.Ps[s] = policy_probs
            self.Vs[s] = valids
            self.Ns[s] = 0
            return v

        # 3. Selection (PUCT)
        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        for a in range(env.action_space_size()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        
        # Execute action
        env.step_flat(a)
        
        # Recursively search
        v = self.search(env)

        # 4. Backpropagation
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v
