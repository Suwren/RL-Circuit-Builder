import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from env.circuit_env import CircuitEnv
from env.components import VoltageSource, Switch, Inductor

def mask_fn(env: gym.Env):
    return env.action_masks()

def train_curriculum():
    # 1. 定义环境
    inventory = [
        VoltageSource(name="Vin", nodes=(0,0), dc_value=12.0), # 0
        Switch(name="S1", nodes=(0,0)),                        # 1
        Switch(name="S2", nodes=(0,0)),                        # 2
        Inductor(name="L1", nodes=(0,0), value=47e-6),         # 3
        VoltageSource(name="Vout", nodes=(0,0), dc_value=5.0)  # 4
    ]
    env = CircuitEnv(initial_components=inventory, max_nodes=6)
    env = ActionMasker(env, mask_fn)
    
    # 2. 定义课程级别 (逆向)
    # 完整动作序列 (Buck):
    # [1, 0, 0, 1] (Vin)
    # [1, 1, 1, 2] (S1)
    # [1, 2, 2, 0] (S2)
    # [1, 3, 2, 3] (L1)
    # [1, 4, 3, 0] (Vout)
    
    full_sequence = [
        [1, 0, 0, 1],
        [1, 1, 1, 2],
        [1, 2, 2, 0],
        [1, 3, 2, 3],
        [1, 4, 3, 0]
    ]
    
    # Level 1: 只差最后一步 (放置 Vout)
    level_1_actions = full_sequence[:-1]
    
    # Level 2: 差最后两步 (放置 L1, Vout)
    level_2_actions = full_sequence[:-2]
    
    # Level 3: 差最后三步 (放置 S2, L1, Vout)
    level_3_actions = full_sequence[:-3]
    
    # Level 4: 从头开始
    level_4_actions = []
    
    levels = [level_1_actions, level_2_actions, level_3_actions, level_4_actions]
    level_names = ["Level 1 (1 step left)", "Level 2 (2 steps left)", "Level 3 (3 steps left)", "Level 4 (Full)"]
    
    # 3. 初始化模型
    model = MaskablePPO("MultiInputPolicy", env, verbose=1, ent_coef=0.01)
    
    # 4. 课程训练循环
    for i, actions in enumerate(levels):
        print(f"\n=== 开始训练 {level_names[i]} ===")
        
        # 设置环境的初始状态
        # 注意: env 是 ActionMasker 包装的，需要 unwrapped
        env.unwrapped.set_initial_actions(actions)
        
        # 训练步数逐渐增加
        timesteps = 10000 * (i + 1)
        
        model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
        model.save(f"ppo_curriculum_level_{i+1}")
        print(f"=== {level_names[i]} 训练完成 ===")

    print("所有课程训练完成！")

if __name__ == "__main__":
    train_curriculum()
