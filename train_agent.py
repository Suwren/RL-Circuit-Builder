import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from env.circuit_env import CircuitEnv
from env.components import VoltageSource, Switch, Inductor
from utils.visualization import visualize_circuit

def train_agent():
    print("正在设置训练环境...")
    
    # 定义元件库存 (同步整流 Buck 变换器所需元件)
    # Vin: 输入电压源 12V
    # S1, S2: 开关 (MOSFET)
    # L1: 电感 47uH
    # Vout: 输出电压源 5V (模拟负载或电池)
    inventory = [
        VoltageSource(name="Vin", nodes=(0,0), dc_value=12.0),
        Switch(name="S1", nodes=(0,0)),
        Switch(name="S2", nodes=(0,0)),
        Inductor(name="L1", nodes=(0,0), value=47e-6),
        VoltageSource(name="Vout", nodes=(0,0), dc_value=5.0)
    ]
    
    # 创建环境实例
    env = CircuitEnv(initial_components=inventory, max_nodes=6)
    
    # 检查环境是否符合 Gym 标准
    print("正在检查环境配置...")
    check_env(env)
    
    # 初始化 PPO (Proximal Policy Optimization) 智能体
    print("正在初始化 PPO 智能体...")
    model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64, ent_coef=0.01)
    
    # 检查是否存在已保存的模型
    import os
    model_path = "ppo_circuit_builder.zip"
    
    if not os.path.exists(model_path):
        print("未找到模型，开始新一轮训练...")
        # 训练模型
        # total_timesteps 决定了训练的总步数，可根据需要调整
        model.learn(total_timesteps=50000)
        print("训练完成。")
        model.save("ppo_circuit_builder")
    else:
        print("发现已有模型，跳过训练直接加载。")
    
    # 加载训练好的模型
    print("正在加载模型...")
    model = PPO.load("ppo_circuit_builder", env=env)
    
    # 评估模型性能
    print("\n正在评估训练后的智能体 (随机模式)...")
    obs, _ = env.reset()
    done = False
    print("正在生成电路...")
    
    while not done:
        # 使用模型预测动作
        # deterministic=False 允许一定的随机性，有助于探索
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 解码并打印动作
        # 动作格式: [类型, 元件索引, 节点1, 节点2]
        act_type = action[0]
        comp_idx = action[1]
        n1 = action[2]
        n2 = action[3]
        
        if act_type == 0:
            print(f"动作: 增加节点 (无效)")
        else:
            comp_name = inventory[comp_idx].name if comp_idx < len(inventory) else "未知"
            print(f"动作: 放置 {comp_name} 于节点 [{n1}, {n2}]")
            
    print(f"最终奖励: {reward}")
    
    # 获取并保存生成的电路连接数据
    edges_data = list(env.circuit_graph.edges(data=True))
    print("电路连接:", edges_data)
    
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write(str(edges_data))
    
    # 可视化生成的电路
    visualize_circuit(env.circuit_graph, "circuit_plot.png")
    print("电路可视化图已保存为 circuit_plot.png")

if __name__ == "__main__":
    train_agent()
