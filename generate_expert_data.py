import gymnasium as gym
import numpy as np
import pickle
from env.circuit_env import CircuitEnv
from env.components import VoltageSource, Switch, Inductor
from sb3_contrib.common.maskable.utils import get_action_masks

def generate_expert_data():
    # 1. 定义环境和库存
    inventory = [
        VoltageSource(name="Vin", nodes=(0,0), dc_value=12.0), # 0
        Switch(name="S1", nodes=(0,0)),                        # 1
        Switch(name="S2", nodes=(0,0)),                        # 2
        Inductor(name="L1", nodes=(0,0), value=47e-6),         # 3
        VoltageSource(name="Vout", nodes=(0,0), dc_value=5.0)  # 4
    ]
    env = CircuitEnv(initial_components=inventory, max_nodes=6)
    obs, _ = env.reset()
    
    # 2. 定义专家动作序列 (标准 Buck 变换器构建)
    # 动作格式: [类型(1), 元件ID, 节点1, 节点2]
    expert_actions = [
        [1, 0, 0, 1], # Place Vin between Node 0 and Node 1 (Creates 0, 1)
        [1, 1, 1, 2], # Place S1 between Node 1 and Node 2 (Creates 2)
        [1, 2, 2, 0], # Place S2 between Node 2 and Node 0
        [1, 3, 2, 3], # Place L1 between Node 2 and Node 3 (Creates 3)
        [1, 4, 3, 0]  # Place Vout between Node 3 and Node 0
    ]
    
    recorded_data = []
    
    print("开始生成专家演示数据...")
    
    for i, action in enumerate(expert_actions):
        # 记录当前观测和专家动作
        # 注意: 我们需要保存 action_masks，因为 MaskablePPO 需要它
        action_masks = get_action_masks(env)
        
        data_point = {
            "observation": obs,
            "action": np.array(action),
            "action_masks": action_masks
        }
        recorded_data.append(data_point)
        
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {i+1}: Action={action}, Reward={reward}, Valid={info['valid_action']}")
        
        if not info['valid_action']:
            print("错误: 专家动作无效！请检查逻辑。")
            return

    print(f"生成完成，共 {len(recorded_data)} 条数据。")
    
    # 3. 保存数据
    with open("expert_data.pkl", "wb") as f:
        pickle.dump(recorded_data, f)
    print("数据已保存至 expert_data.pkl")

if __name__ == "__main__":
    generate_expert_data()
