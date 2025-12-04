import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from copy import deepcopy

from env.components import Component, Inductor, Capacitor, Resistor, VoltageSource, Switch, Diode, Wire

class CircuitEnv(gym.Env):
    """
    电力电子电路构建环境 (OpenAI Gym 接口)。
    
    该环境允许智能体通过放置元件来构建电路拓扑。
    目标是构建一个功能性的电源变换器（如 Buck 变换器）。
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, initial_components: List[Component], max_nodes: int = 20):
        """
        初始化环境。
        
        参数:
            initial_components: 可用元件的清单 (库存)。
            max_nodes: 电路中允许的最大节点数。
        """
        super().__init__()
        self.max_nodes = max_nodes
        # 深拷贝以确保环境拥有独立的元件列表
        self.initial_inventory = deepcopy(initial_components)
        self.max_components = len(initial_components)
        
        # 电路的图表示 (使用 NetworkX MultiGraph 支持多重边)
        self.circuit_graph = nx.MultiGraph()
        self.node_counter = 1 
        
        # 动作空间 (Action Space)
        # 格式: [动作类型, 元件索引, 节点1, 节点2]
        # 动作类型: 0=增加节点 (已禁用), 1=放置元件
        self.action_space = spaces.MultiDiscrete([2, self.max_components, max_nodes, max_nodes])

        # 观测空间 (Observation Space)
        # 包含:
        # 1. adjacency: 邻接矩阵，表示电路连接结构
        # 2. inventory_mask: 库存掩码，表示哪些元件尚未使用 (1=可用, 0=已用)
        # 3. node_features: 节点特征 (预留，目前全零)
        self.observation_space = spaces.Dict({
            "adjacency": spaces.Box(low=0, high=1, shape=(max_nodes, max_nodes), dtype=np.int8),
            "inventory_mask": spaces.Box(low=0, high=1, shape=(self.max_components,), dtype=np.int8),
            "node_features": spaces.Box(low=-np.inf, high=np.inf, shape=(max_nodes, 2), dtype=np.float32),
        })

        self.reset()

    def reset(self, seed=None, options=None):
        """
        重置环境到初始状态。
        """
        super().reset(seed=seed)
        self.circuit_graph.clear()
        
        # 添加接地节点 (GND)
        self.circuit_graph.add_node(0, type="GND") 
        self.node_counter = 1
        
        # 预先添加所有可能的中间节点，简化动作空间
        for _ in range(self.max_nodes - 1):
            self.circuit_graph.add_node(self.node_counter, type="Intermediate")
            self.node_counter += 1
            
        self.step_count = 0
        self.last_cycle_count = 0 # 初始化环路计数，用于奖励计算
        
        # 设置最大步数，给予一定的冗余
        self.max_steps = self.max_components * 3 
        if self.max_steps < 10: self.max_steps = 10
        
        # 重置库存状态 (1 = 可用)
        self.available_components = np.ones(self.max_components, dtype=np.int8)
        
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        """
        执行一步动作。
        """
        self.step_count += 1
        
        # 1. 执行动作 (放置元件)
        valid_action = self._apply_action(action)
        
        # 2. 计算奖励
        # 仿真和分析逻辑集成在 _calculate_reward 中
        simulation_result = None
        reward = self._calculate_reward(simulation_result, valid_action, action)
        
        # 3. 检查终止条件
        terminated = False
        # 如果所有元件都已使用，任务完成
        if np.sum(self.available_components) == 0:
            terminated = True
            
        truncated = False
        # 如果达到最大步数
        if self.step_count >= self.max_steps:
            truncated = True
            # 惩罚：如果因超时结束且仍有元件未用，给予惩罚
            if np.sum(self.available_components) > 0:
                reward -= 10.0 
        
        # 4. 获取新的观测
        observation = self._get_obs()
        info = {}
        
        return observation, reward, terminated, truncated, info

    def _apply_action(self, action) -> bool:
        """
        应用动作到电路图中。
        返回动作是否有效。
        """
        # 解析动作: [类型, 元件索引, 节点1, 节点2]
        action_type = action[0]
        comp_idx = action[1]
        n1 = action[2]
        n2 = action[3]
        
        if action_type == 0: # 增加节点 (已禁用)
            return False
                
        elif action_type == 1: # 放置元件
            # 检查元件索引是否有效且库存中有剩余
            if comp_idx < self.max_components and self.available_components[comp_idx] == 1:
                # 检查节点是否存在且不是同一个节点 (不能短接自身)
                if self.circuit_graph.has_node(n1) and self.circuit_graph.has_node(n2) and n1 != n2:
                    # 从初始库存获取元件模板
                    comp_template = self.initial_inventory[comp_idx]
                    # 创建元件副本
                    comp_instance = deepcopy(comp_template)
                    comp_instance.nodes = (n1, n2)
                    
                    # 将元件作为边添加到图中
                    self.circuit_graph.add_edge(n1, n2, component=comp_instance, inventory_idx=comp_idx)
                    
                    # 标记该元件为已使用
                    self.available_components[comp_idx] = 0
                    return True
        
        return False

    def _calculate_reward(self, result, valid_action, action):
        """
        计算奖励函数。
        基于论文策略：过程奖励 (环路形成) + 终局奖励 (功能验证)。
        """
        # 奖励参数定义
        R_SN = -1.0   # 小惩罚: 未形成新环路
        R_MN = -10.0  # 中惩罚: 形成无效环路 / 严重错误
        R_SP = +2.0   # 小奖励: 形成有效新环路
        R_MP = +10.0  # 中奖励: 所有电源都在环路中
        R_BP = +100.0 # 大奖励: 功能验证通过 (伏秒平衡)
        
        action_type = action[0]

        # --- 1. 过程奖励 (Step Rewards) ---
        
        if not valid_action:
            return -0.5 # 无效动作给予微小惩罚
            
        reward = 0.0
        
        # 检查是否形成了新环路
        try:
            # 获取当前的环路基 (Cycle Basis)
            # 注意: cycle_basis 返回的是无向图的基础环列表
            current_cycles = nx.cycle_basis(self.circuit_graph.to_undirected())
            num_cycles = len(current_cycles)
        except:
            num_cycles = 0
            current_cycles = []
            
        if num_cycles > self.last_cycle_count:
            # 发现了新环路
            has_bad_loop = False
            
            # 检查所有环路的有效性
            for cycle_nodes in current_cycles:
                loop_components = []
                # 获取环路中的所有元件
                for i in range(len(cycle_nodes)):
                    u = cycle_nodes[i]
                    v = cycle_nodes[(i + 1) % len(cycle_nodes)]
                    if self.circuit_graph.has_edge(u, v):
                        # 获取边上的元件 (简化: 取第一个)
                        edges = self.circuit_graph.get_edge_data(u, v)
                        for key, data in edges.items():
                            loop_components.append(data['component'])
                            break 
                            
                # 规则 2a: 环路中必须包含电感 (防止电压源短路)
                has_inductor = any(isinstance(c, Inductor) for c in loop_components)
                
                # 规则 2b: 环路中必须包含开关或二极管 (保证可控性)
                has_switch_or_diode = any(isinstance(c, (Switch, Diode)) for c in loop_components)
                
                if not has_inductor:
                    # 潜在短路风险 (如 Vin-Switch-GND)
                    has_bad_loop = True
                    
                if not has_switch_or_diode:
                    # 不可控回路 (如 Vin-L-Vout 直通)
                    has_bad_loop = True
            
            if has_bad_loop:
                reward += R_MN # 发现坏环路，惩罚
            else:
                reward += R_SP # 发现好环路，奖励
                
        else:
            # 没有发现新环路
            reward += R_SN # 鼓励尽快闭合回路
            
        self.last_cycle_count = num_cycles
        
        # 额外检查: 并联开关惩罚
        # 智能体不应在两点间并联多个开关
        n1, n2 = action[2], action[3]
        if self.circuit_graph.has_edge(n1, n2):
            edges = self.circuit_graph.get_edge_data(n1, n2)
            switch_count = sum(1 for k, d in edges.items() if isinstance(d['component'], Switch))
            if switch_count > 1:
                reward += R_MN # 严厉惩罚并联开关
        
        # --- 2. 终局奖励 (Terminal Rewards) ---
        # 当所有元件放置完毕时触发
        if np.sum(self.available_components) == 0:
            
            # 检查 1: 所有电压源是否都在环路中
            all_sources_in_loops = True
            sources = [d['component'] for u, v, d in self.circuit_graph.edges(data=True) if isinstance(d['component'], VoltageSource)]
            
            # 构建所有环路的边集合
            cycle_edges = set()
            for cycle in current_cycles:
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    cycle_edges.add(tuple(sorted((u, v))))
            
            for src in sources:
                u, v = src.nodes
                if tuple(sorted((u, v))) not in cycle_edges:
                    all_sources_in_loops = False
                    break
            
            if all_sources_in_loops and len(sources) > 0:
                reward += R_MP # 奖励: 电源有效接入
            
            # 检查 2: 功能性验证 (伏秒平衡)
            from utils.mode_analysis import analyze_switching_modes
            
            # 首先检查全图连通性
            if not nx.is_connected(self.circuit_graph.to_undirected()):
                reward += R_MN # 不连通，无效
                return reward
                
            # 调用模态分析工具
            modes = analyze_switching_modes(self.circuit_graph)
            
            has_short = False
            has_increasing = False
            has_decreasing = False
            
            for state, data in modes.items():
                if not data['valid']:
                    has_short = True # 发现短路或断路模态
                
                if data['valid']:
                    # 检查电感电流趋势
                    trends = data['inductor_trends']
                    for comp_name, trend in trends.items():
                        if "Increasing" in trend: has_increasing = True
                        if "Decreasing" in trend: has_decreasing = True
            
            if has_short:
                # 存在严重故障 (短路/断路)
                reward += R_MN 
            elif has_increasing and has_decreasing:
                # 成功: 电感电流既能增加也能减少 (满足伏秒平衡条件)
                reward += R_BP # 给予最大奖励!
            else:
                # 失败: 电路无法正常工作 (如只有单向电流)
                reward += R_MN
                
        return reward

    def _get_obs(self):
        """
        生成当前的观测向量。
        """
        # 1. 邻接矩阵 (Adjacency Matrix)
        adj = np.zeros((self.max_nodes, self.max_nodes), dtype=np.int8)
        for u, v in self.circuit_graph.edges():
            if u < self.max_nodes and v < self.max_nodes:
                adj[u, v] = 1
                adj[v, u] = 1
        
        return {
            "adjacency": adj,
            "inventory_mask": self.available_components.copy(),
            "node_features": np.zeros((self.max_nodes, 2), dtype=np.float32), # 预留
        }

    def render(self):
        pass
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, initial_components: List[Component], max_nodes: int = 20):
        super().__init__()
        self.max_nodes = max_nodes
        # 深拷贝以确保环境拥有独立的元件列表
        self.initial_inventory = deepcopy(initial_components)
        self.max_components = len(initial_components)
        
        # 电路的图表示
        self.circuit_graph = nx.MultiGraph()
        self.node_counter = 1 
        
        # 动作空间 (Action Space)
        # [类型, 元件索引, 节点1, 节点2]
        # 类型: 0=增加节点, 1=放置元件
        self.action_space = spaces.MultiDiscrete([2, self.max_components, max_nodes, max_nodes])

        # 观测空间 (Observation Space)
        # 我们需要表示:
        # 1. 电路图 (邻接矩阵)
        # 2. 库存状态 (哪些元件已被使用)
        self.observation_space = spaces.Dict({
            "adjacency": spaces.Box(low=0, high=1, shape=(max_nodes, max_nodes), dtype=np.int8),
            "inventory_mask": spaces.Box(low=0, high=1, shape=(self.max_components,), dtype=np.int8), # 1=可用, 0=已用
            "node_features": spaces.Box(low=-np.inf, high=np.inf, shape=(max_nodes, 2), dtype=np.float32),
        })

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.circuit_graph.clear()
        self.circuit_graph.add_node(0, type="GND") 
        self.node_counter = 1
        
        # 预先添加所有节点以简化动作空间
        for _ in range(self.max_nodes - 1):
            self.circuit_graph.add_node(self.node_counter, type="Intermediate")
            self.node_counter += 1
            
        self.step_count = 0
        self.last_cycle_count = 0 # Initialize cycle count for reward calculation
        self.max_steps = self.max_components * 3 # 允许一些额外的步骤
        if self.max_steps < 10: self.max_steps = 10
        
        # 重置库存
        # 我们跟踪初始列表中哪些元件当前在图中
        # 为了简单起见，我们只保留一个布尔掩码
        self.available_components = np.ones(self.max_components, dtype=np.int8)
        
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        self.step_count += 1
        
        # 1. 执行动作
        valid_action = self._apply_action(action)
        
        # 2. 仿真 (已移除实时仿真，改为终局分析)
        simulation_result = None
        
        # 3. 计算奖励
        reward = self._calculate_reward(simulation_result, valid_action, action)
        
        # 4. 检查终止条件
        terminated = False
        if np.sum(self.available_components) == 0:
            terminated = True
            
        truncated = False
        if self.step_count >= self.max_steps:
            truncated = True
            # 如果因步数限制而终止，且还有剩余元件，给予惩罚
            if np.sum(self.available_components) > 0:
                reward -= 10.0 # 未完成惩罚
        
        # 5. 获取观测
        observation = self._get_obs()
        info = {}
        
        return observation, reward, terminated, truncated, info

    def _apply_action(self, action) -> bool:
        # 动作现在是一个数组: [类型, 元件索引, 节点1, 节点2]
        action_type = action[0]
        comp_idx = action[1]
        n1 = action[2]
        n2 = action[3]
        
        if action_type == 0: # 增加节点
            # 已禁用: 节点已预先添加
            return False
                
        elif action_type == 1: # 放置元件
            # 检查元件是否可用
            if comp_idx < self.max_components and self.available_components[comp_idx] == 1:
                # 检查节点是否存在且不同
                if self.circuit_graph.has_node(n1) and self.circuit_graph.has_node(n2) and n1 != n2:
                    # 从初始库存获取元件模板
                    comp_template = self.initial_inventory[comp_idx]
                    # 创建副本/实例
                    comp_instance = deepcopy(comp_template)
                    comp_instance.nodes = (n1, n2)
                    
                    # 添加到图
                    self.circuit_graph.add_edge(n1, n2, component=comp_instance, inventory_idx=comp_idx)
                    
                    # 标记为已使用
                    self.available_components[comp_idx] = 0
                    return True
        
        return False

    def _calculate_reward(self, result, valid_action, action):
        # 论文奖励参数
        R_SN = -1.0   # Small Negative (No new loop)
        R_MN = -10.0  # Medium Negative (Invalid loop)
        R_BN = -20.0  # Big Negative (Timeout/Redundant - handled elsewhere)
        R_SP = +2.0   # Small Positive (Valid new loop)
        R_MP = +10.0  # Medium Positive (All sources in loops)
        R_BP = +100.0 # Big Positive (Volt-Second Balance / Functional)
        
        # 动作: [类型, 元件索引, 节点1, 节点2]
        action_type = action[0]

        # 1. 过程奖励 (Step Rewards)
        if not valid_action:
            return -0.5 # 无效动作微小惩罚
            
        reward = 0.0
        
        # 检查是否形成了新环路
        try:
            # 使用 simple_cycles 对于有向图，但我们是无向图 (电气连接)
            # cycle_basis 返回基础环
            current_cycles = nx.cycle_basis(self.circuit_graph.to_undirected())
            num_cycles = len(current_cycles)
        except:
            num_cycles = 0
            current_cycles = []
            
        if num_cycles > self.last_cycle_count:
            # 发现了新环路
            # 检查新环路的性质 (简化：检查所有环路，如果有坏环路则惩罚)
            # 理想情况下应该只检查新生成的，但 cycle_basis 顺序不确定。
            # 我们检查是否存在"坏环路"。
            
            has_bad_loop = False
            
            for cycle_nodes in current_cycles:
                # 获取环路中的元件
                loop_components = []
                # 遍历环路节点对 (u, v)
                for i in range(len(cycle_nodes)):
                    u = cycle_nodes[i]
                    v = cycle_nodes[(i + 1) % len(cycle_nodes)]
                    # 获取 u, v 之间的边数据
                    if self.circuit_graph.has_edge(u, v):
                        # 可能有多条边，我们只取第一条 (简化)
                        # 严格来说应该检查所有路径组合，但这里假设 cycle_basis 对应物理路径
                        edges = self.circuit_graph.get_edge_data(u, v)
                        for key, data in edges.items():
                            loop_components.append(data['component'])
                            break # 只取一个
                            
                # 规则 2a: 环路中必须包含电感 L (防止短路)
                has_inductor = any(isinstance(c, Inductor) for c in loop_components)
                
                # 规则 2b: 环路中必须包含开关 S 或二极管 D (可控性)
                has_switch_or_diode = any(isinstance(c, (Switch, Diode)) for c in loop_components)
                
                if not has_inductor:
                    # 不包含电感 -> 潜在短路风险 (如 Vin-S-GND)
                    # 除非是纯电阻回路 (但我们没有纯电阻负载，只有 Vout)
                    # Vout 也是电压源，所以 Vin-Vout 也是短路
                    has_bad_loop = True
                    # print("Penalty: Loop without Inductor detected")
                    
                if not has_switch_or_diode:
                    # 不包含开关 -> 不可控 (如 Vin-L-Vout 直接导通)
                    has_bad_loop = True
                    # print("Penalty: Uncontrollable Loop detected")
            
            if has_bad_loop:
                reward += R_MN
            else:
                reward += R_SP
                
        else:
            # 没有发现新环路
            reward += R_SN
            
        self.last_cycle_count = num_cycles
        
        # 检查并联开关 (保留此规则，虽然论文没明确说，但属于"冗余"范畴)
        n1, n2 = action[2], action[3]
        if self.circuit_graph.has_edge(n1, n2):
            edges = self.circuit_graph.get_edge_data(n1, n2)
            switch_count = sum(1 for k, d in edges.items() if isinstance(d['component'], Switch))
            if switch_count > 1:
                reward += R_MN # 使用中等惩罚
        
        # 检查库存是否为空 (终止状态)
        if np.sum(self.available_components) == 0:
            # 终局奖励
            
            # 1. 检查所有电压源是否都在环路中 (R_MP)
            # 简单检查：电压源所在的边是否属于某个 cycle
            # cycle_basis 返回节点列表。
            all_sources_in_loops = True
            sources = [d['component'] for u, v, d in self.circuit_graph.edges(data=True) if isinstance(d['component'], VoltageSource)]
            
            # 构建所有环路的边集合
            cycle_edges = set()
            for cycle in current_cycles:
                for i in range(len(cycle)):
                    u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                    cycle_edges.add(tuple(sorted((u, v))))
            
            for src in sources:
                u, v = src.nodes
                if tuple(sorted((u, v))) not in cycle_edges:
                    all_sources_in_loops = False
                    break
            
            if all_sources_in_loops and len(sources) > 0:
                reward += R_MP
            
            # 2. 功能性检查 (R_BP - 伏秒平衡/可运行)
            from utils.mode_analysis import analyze_switching_modes
            
            # 首先检查连通性
            if not nx.is_connected(self.circuit_graph.to_undirected()):
                reward += R_MN # 不连通视为不可控/无效
                return reward
                
            # 分析模态
            modes = analyze_switching_modes(self.circuit_graph)
            
            has_short = False
            has_increasing = False
            has_decreasing = False
            
            for state, data in modes.items():
                if not data['valid']:
                    has_short = True
                
                if data['valid']:
                    trends = data['inductor_trends']
                    for comp_name, trend in trends.items():
                        if "Increasing" in trend: has_increasing = True
                        if "Decreasing" in trend: has_decreasing = True
            
            if has_short:
                # 严重故障
                reward += R_MN # 或者更重
            elif has_increasing and has_decreasing:
                # 满足伏秒平衡条件 (既能增磁也能消磁)
                reward += R_BP
            else:
                # 不满足
                reward += R_MN
                
        return reward

    def _get_obs(self):
        # 构建邻接矩阵
        adj = np.zeros((self.max_nodes, self.max_nodes), dtype=np.int8)
        
        for u, v in self.circuit_graph.edges():
            if u < self.max_nodes and v < self.max_nodes:
                adj[u, v] = 1
                adj[v, u] = 1
        
        return {
            "adjacency": adj,
            "inventory_mask": self.available_components.copy(),
            "node_features": np.zeros((self.max_nodes, 2), dtype=np.float32),
        }

    def render(self):
        pass


