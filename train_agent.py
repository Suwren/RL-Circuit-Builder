import gymnasium as gym
from typing import List
import warnings
from stable_baselines3.common.env_checker import check_env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from env.circuit_env import CircuitEnv
from env.components import VoltageSource, Switch, Inductor
from utils.visualization import visualize_circuit

# 过滤 Stable Baselines3 的特定警告 (关于观测空间形状)
warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

def mask_fn(env: gym.Env) -> List[bool]:
    return env.action_masks()

from stable_baselines3.common.callbacks import BaseCallback

class EntropyDecayCallback(BaseCallback):
    """
    自定义回调函数，用于在训练过程中动态调整熵系数 (ent_coef)。
    实现从 initial_ent_coef 线性衰减到 final_ent_coef。
    """
    def __init__(self, initial_ent_coef: float, final_ent_coef: float, decay_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.decay_steps = decay_steps
        self.current_ent_coef = initial_ent_coef

    def _on_step(self) -> bool:
        # 计算当前的进度 (0.0 到 1.0)
        progress = min(1.0, self.num_timesteps / self.decay_steps)
        
        # 线性插值计算新的 ent_coef
        new_ent_coef = self.initial_ent_coef + progress * (self.final_ent_coef - self.initial_ent_coef)
        
        # 更新模型的 ent_coef
        self.model.ent_coef = new_ent_coef
        self.current_ent_coef = new_ent_coef
        
        # 定期打印日志
        if self.verbose > 0 and self.n_calls % 1000 == 0:
            print(f"Step {self.num_timesteps}: Entropy Coef updated to {new_ent_coef:.4f}")
            
        return True

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
    # 使用 ActionMasker 包装环境
    env = ActionMasker(env, mask_fn)
    
    # 检查环境是否符合 Gym 标准
    print("正在检查环境配置...")
    # check_env 可能会产生警告，已被上方 filterwarnings 屏蔽
    # check_env(env) # ActionMasker 可能不完全兼容 check_env 的某些检查，暂时跳过或忽略
    
    # 初始化 MaskablePPO 智能体
    print("正在初始化 MaskablePPO 智能体...")
    model = MaskablePPO("MultiInputPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64, ent_coef=0.05)
    
    # 检查是否存在已保存的模型
    import os
    model_path = "ppo_circuit_builder.zip"
    
    FORCE_RETRAIN = True # 强制重新训练标志
    
    if FORCE_RETRAIN or not os.path.exists(model_path):
        if FORCE_RETRAIN:
            print("强制重新训练模式已开启，忽略现有模型...")
        else:
            print("未找到模型，开始新一轮训练...")
            
        # 初始化熵衰减回调
        # 从 0.2 (高随机性) 衰减到 0.01 (低随机性)，在前 30,000 步完成衰减
        entropy_callback = EntropyDecayCallback(initial_ent_coef=0.2, final_ent_coef=0.01, decay_steps=30000, verbose=1)
        
        # 训练模型
        # total_timesteps 决定了训练的总步数，可根据需要调整
        model.learn(total_timesteps=50000, callback=entropy_callback)
        print("训练完成。")
        model.save("ppo_circuit_builder")
    else:
        print("发现已有模型，跳过训练直接加载。")
    
    # 加载训练好的模型
    print("正在加载模型...")
    model = MaskablePPO.load("ppo_circuit_builder", env=env)
    
    # 评估模型性能
    print("\n正在评估训练后的智能体 (随机模式)...")
    obs, _ = env.reset()
    done = False
    print("正在生成电路...")
    
    from sb3_contrib.common.maskable.utils import get_action_masks
    
    while not done:
        # 获取当前状态的动作掩码
        action_masks = get_action_masks(env)
        
        # 使用模型预测动作，必须传入 action_masks
        # deterministic=False 允许一定的随机性，有助于探索
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 解码并打印动作
        # 动作格式: [类型, 元件索引, 节点1, 节点2]
        act_type = action[0]
        comp_idx = action[1]
        n1 = action[2]
        n2 = action[3]
        
        # 获取动作有效性
        is_valid = info.get("valid_action", False)
        status_str = "(成功)" if is_valid else "(无效/失败)"
        
        if act_type == 0:
            print(f"动作: 增加节点 {status_str}")
        else:
            comp_name = inventory[comp_idx].name if comp_idx < len(inventory) else "未知"
            print(f"动作: 放置 {comp_name} 于节点 [{n1}, {n2}] {status_str}")
            
    print(f"最终奖励: {reward}")
    
    # 获取并保存生成的电路连接数据
    # 注意: env 被 ActionMasker 包装，需要解包获取原始环境
    raw_env = env.unwrapped
    edges_data = list(raw_env.circuit_graph.edges(data=True))
    print("电路连接:", edges_data)
    
    with open("result.txt", "w", encoding="utf-8") as f:
        f.write(str(edges_data))
    
    # 可视化生成的电路
    visualize_circuit(raw_env.circuit_graph, "circuit_plot.png")
    print("电路可视化图已保存为 circuit_plot.png")

if __name__ == "__main__":
    train_agent()
