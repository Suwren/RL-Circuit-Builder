from env.circuit_env import CircuitEnv
from env.components import VoltageSource, Switch, Inductor

def test_empty_penalty():
    print("Testing Empty Circuit Penalty...")
    
    # Inventory: V1, S1, L1
    inventory = [
        VoltageSource("V1", (0,0), 10.0),
        Switch("S1", (0,0)),
        Inductor("L1", (0,0), 100e-6)
    ]
    
    env = CircuitEnv(inventory, max_nodes=5, verbose=True)
    
    # Do nothing, just calculate score immediately (simulating immediate STOP)
    print("Calculating Score for Empty Circuit (Expect -100 penalty)...")
    score = env._calculate_circuit_score()
    print(f"Final Score: {score}")
    
    if score <= -100:
        print("Test PASSED: Penalty applied.")
    else:
        print("Test FAILED: Penalty NOT applied.")

if __name__ == "__main__":
    test_empty_penalty()
