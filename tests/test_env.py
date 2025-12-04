import gymnasium as gym
import numpy as np
from env.circuit_env import CircuitEnv
from env.components import Inductor, Capacitor, VoltageSource, Switch, Diode

def test_env():
    print("Initializing Environment with Inventory...")
    
    # ==========================================
    # 定义元件库存 (Component Inventory)
    # ==========================================
    # 这里定义了智能体可以使用的所有元件。
    # 您可以自由修改这个列表来改变智能体可用的“积木”。
    # 
    # 参数说明:
    # - name: 元件名称 (用于网表生成，需唯一)
    # - nodes: 初始节点 (在环境中会被动作覆盖，这里填 (0,0) 即可)
    # - value: 元件参数 (如电感值、电容值、电阻值)
    # - dc_value: 电压源的直流电压值
    # 
    # 示例:
    # 要增加一个电感，只需在列表中添加: Inductor(name="L2", nodes=(0,0), value=10e-6)
    inventory = [
        VoltageSource(name="Vin", nodes=(0,0), dc_value=10.0), # 输入电压源 10V
        Inductor(name="L1", nodes=(0,0), value=10e-6),         # 电感 10uH
        Switch(name="S1", nodes=(0,0)),                        # 开关
        Diode(name="D1", nodes=(0,0)),                         # 二极管
        Capacitor(name="C1", nodes=(0,0), value=22e-6)         # 电容 22uF
    ]
    
    # 将库存传递给环境
    env = CircuitEnv(initial_components=inventory)
    
    print("Resetting Environment...")
    obs, info = env.reset()
    print("Observation Keys:", obs.keys())
    print("Inventory Mask:", obs["inventory_mask"])
    
    # Test Add Node
    print("\nTesting Add Node...")
    action = {
        "type": 0, # Add Node
        "component_idx": 0, # Ignored
        "nodes": np.array([0, 0]) # Ignored
    }
    obs, reward, term, trunc, info = env.step(action)
    print("Nodes:", env.circuit_graph.nodes())
    
    # Test Place Component (Inductor L1 between Node 1 and Node 2)
    # First add another node so we have 0, 1, 2
    env.step(action) 
    print("Nodes:", env.circuit_graph.nodes())
    
    print("\nTesting Place Component (L1 between 1 and 2)...")
    action = {
        "type": 1, # Place Component
        "component_idx": 1, # L1 is at index 1
        "nodes": np.array([1, 2])
    }
    obs, reward, term, trunc, info = env.step(action)
    print("Edges:", env.circuit_graph.edges(data=True))
    print("Inventory Mask after placement:", obs["inventory_mask"])
    
    # Test Invalid Placement (Reuse L1)
    print("\nTesting Invalid Placement (Reuse L1)...")
    obs, reward, term, trunc, info = env.step(action)
    print("Reward for invalid action:", reward)
    
    print("\nEnvironment Test Complete!")

if __name__ == "__main__":
    try:
        test_env()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
