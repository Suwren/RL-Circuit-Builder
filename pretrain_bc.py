import gymnasium as gym
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from sb3_contrib import MaskablePPO
from env.circuit_env import CircuitEnv
from env.components import VoltageSource, Switch, Inductor
from sb3_contrib.common.wrappers import ActionMasker

def mask_fn(env: gym.Env):
    return env.action_masks()

def pretrain_bc():
    # 1. 加载专家数据
    with open("expert_data.pkl", "rb") as f:
        expert_data = pickle.load(f)
    print(f"加载了 {len(expert_data)} 条专家演示数据。")
    
    # 2. 初始化环境和模型
    inventory = [
        VoltageSource(name="Vin", nodes=(0,0), dc_value=12.0),
        Switch(name="S1", nodes=(0,0)),
        Switch(name="S2", nodes=(0,0)),
        Inductor(name="L1", nodes=(0,0), value=47e-6),
        VoltageSource(name="Vout", nodes=(0,0), dc_value=5.0)
    ]
    env = CircuitEnv(initial_components=inventory, max_nodes=6)
    env = ActionMasker(env, mask_fn)
    
    model = MaskablePPO("MultiInputPolicy", env, verbose=1)
    
    # 3. 提取策略网络
    policy = model.policy
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 4. 行为克隆训练循环
    epochs = 100
    print(f"开始行为克隆预训练 ({epochs} epochs)...")
    
    for epoch in range(epochs):
        total_loss = 0
        for data in expert_data:
            obs = data["observation"]
            action = data["action"]
            action_masks = data["action_masks"]
            
            # 转换观测为 Tensor
            # SB3 的 predict 需要处理过的观测，这里我们手动处理一下
            # 或者直接使用 policy.obs_to_tensor
            obs_tensor, _ = policy.obs_to_tensor(obs)
            
            # 获取策略分布
            # MaskablePPO 的 forward 返回 (actions, values, log_probs)
            # 我们需要分布 logits 来计算 CrossEntropy
            # policy.get_distribution(obs_tensor, action_masks)
            
            # 注意: MaskablePPO 的分布比较复杂 (MultiDiscrete)
            # 我们需要分别计算每个维度的 Loss
            
            distribution = policy.get_distribution(obs_tensor, torch.as_tensor(action_masks).unsqueeze(0))
            
            # 获取 Logits (对于 MultiDiscrete，这是一个 Categorical 分布的列表)
            # SB3 的 Distribution 并不直接暴露 logits，但我们可以通过 log_prob 反推，或者直接采样
            # 更简单的方法: 最大化专家动作的 log_prob (即最小化 NLL Loss)
            
            action_tensor = torch.as_tensor(action).unsqueeze(0).to(policy.device)
            log_prob = distribution.log_prob(action_tensor)
            
            loss = -log_prob.mean() # Negative Log Likelihood
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
            
    # 5. 保存预训练模型
    model.save("ppo_circuit_builder")
    print("预训练模型已保存为 ppo_circuit_builder.zip")

if __name__ == "__main__":
    pretrain_bc()
