from env.circuit_env import CircuitEnv
from env.components import VoltageSource, Switch, Inductor

def test_unused_penalty():
    print("Testing Unused Component Penalty...")
    
    # Inventory: V1, S1, L1
    inventory = [
        VoltageSource("V1", (0,0), 10.0),
        Switch("S1", (0,0)),
        Inductor("L1", (0,0), 100e-6)
    ]
    
    env = CircuitEnv(inventory, max_nodes=5, verbose=True)
    
    # Place V1 and S1, but leave L1 unused
    # V1: 0-1
    env._add_component_to_graph(0, 0, 1)
    # S1: 1-0
    env._add_component_to_graph(1, 1, 0)
    
    # Mark them as used in available_components
    env.available_components[0] = 0
    env.available_components[1] = 0
    # L1 (index 2) remains 1 (available/unused)
    
    print("Calculating Score (Expect -100 penalty for unused L1)...")
    score = env._calculate_circuit_score()
    print(f"Final Score: {score}")
    
    if score <= -100:
        print("Test PASSED: Penalty applied.")
    else:
        print("Test FAILED: Penalty NOT applied.")

if __name__ == "__main__":
    test_unused_penalty()
