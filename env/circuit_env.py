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
    
    **新动作逻辑 (Blank Canvas)**:
    1. 初始为空白。
    2. 第一个元件强制创建节点 0 和 1。
    3. 后续元件必须至少有一个端点连接到现有节点。
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, initial_components: List[Component], max_nodes: int = 20):
        """
        初始化环境。
        
        参数:
            initial_components: 可用元件的清单 (库存)。
            max_nodes: 电路中允许的最大节点数 (用于观测空间的大小限制)。
        """
        super().__init__()
        self.max_nodes = max_nodes
        # 深拷贝以确保环境拥有独立的元件列表
        self.initial_inventory = deepcopy(initial_components)
        self.max_components = len(initial_components)
        
        # 电路的图表示 (使用 NetworkX MultiGraph 支持多重边)
        self.circuit_graph = nx.MultiGraph()
        self.node_counter = 0 
        
        # 动作空间 (Action Space)
        # 格式: [动作类型, 元件索引, 节点1, 节点2]
        # 动作类型: 0=增加节点 (禁用), 1=放置元件, 2=重连元件 (Rewire)
        self.action_space = spaces.MultiDiscrete([3, self.max_components, max_nodes, max_nodes])

        # 观测空间 (Observation Space)
        # 包含:
        # 1. adjacency: 邻接矩阵，表示电路连接结构
        # 2. inventory_mask: 库存掩码，表示哪些元件尚未使用 (1=可用, 0=已用)
        # 3. node_features: 节点特征 (预留，目前全零)
        self.observation_space = spaces.Dict({
            "adjacency": spaces.Box(low=0, high=9, shape=(max_nodes, max_nodes), dtype=np.int8),
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
        
        # 初始状态：无节点，无元件
        self.node_counter = 0
            
        self.step_count = 0
        self.last_cycle_count = 0 # 初始化环路计数，用于奖励计算
        
        # 设置最大步数，给予一定的冗余
        # 增加步数以允许重连
        self.max_steps = self.max_components * 5 
        if self.max_steps < 20: self.max_steps = 20
        
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
        
        # 1. 执行动作 (放置元件 或 重连)
        valid_action = self._apply_action(action)
        
        # 2. 计算奖励
        # 仿真和分析逻辑集成在 _calculate_reward 中
        simulation_result = None
        reward = self._calculate_reward(simulation_result, valid_action, action)
        
        # 3. 检查终止条件
        terminated = False
        
        # 只有当所有元件都已使用，且达到一定条件(如分数很高)或步数耗尽时才结束
        # 但为了简化，我们允许在所有元件用完后继续重连，直到 max_steps
        # 或者增加一个 "停止" 动作? 
        # 这里简化：只要步数没到，且所有元件都用了，就继续跑 (进入重连阶段)
        
        truncated = False
        # 如果达到最大步数
        if self.step_count >= self.max_steps:
            truncated = True
            # 惩罚：如果因超时结束且仍有元件未用，给予惩罚
            if np.sum(self.available_components) > 0:
                reward -= 10.0 
        
        # 4. 获取新的观测
        observation = self._get_obs()
        info = {
            "valid_action": valid_action,
            "simulation_result": simulation_result
        }
        
        return observation, reward, terminated, truncated, info

    def _apply_action(self, action) -> bool:
        """
        应用动作到电路图中。
        遵循"白纸"构建逻辑。
        返回动作是否有效。
        """
        # 解析动作: [类型, 元件索引, 节点1, 节点2]
        action_type = action[0]
        comp_idx = action[1]
        raw_n1 = action[2]
        raw_n2 = action[3]
        
        if action_type == 0: # 增加节点 (已禁用)
            return False
                
        elif action_type == 1: # 放置元件
            # 检查元件索引是否有效且库存中有剩余
            if comp_idx < self.max_components and self.available_components[comp_idx] == 1:
                
                # --- 逻辑分支 1: 放置第一个元件 ---
                if self.circuit_graph.number_of_nodes() == 0:
                    # 强制创建节点 0 和 1
                    n1, n2 = 0, 1
                    self.circuit_graph.add_node(n1, type="Intermediate")
                    self.circuit_graph.add_node(n2, type="Intermediate")
                    self.node_counter = 2
                    
                    # 添加元件
                    self._add_component_to_graph(comp_idx, n1, n2)
                    return True
                
                # --- 逻辑分支 2: 放置后续元件 ---
                else:
                    current_node_count = self.node_counter
                    
                    is_n1_new = raw_n1 >= current_node_count
                    is_n2_new = raw_n2 >= current_node_count
                    
                    # 约束检查: 必须至少有一个连接到现有节点
                    if is_n1_new and is_n2_new:
                        return False # 试图创建孤岛
                    
                    # 确定最终的节点 ID
                    final_n1 = raw_n1 if not is_n1_new else current_node_count
                    
                    if is_n2_new:
                        final_n2 = current_node_count
                    else:
                        final_n2 = raw_n2
                        
                    # 再次检查不能连接同一个节点
                    if final_n1 == final_n2:
                        return False
                    
                    # 执行添加
                    # 如果需要创建新节点
                    if is_n1_new or is_n2_new:
                        # 检查是否超过最大节点限制
                        if self.node_counter >= self.max_nodes:
                            return False
                        
                        self.circuit_graph.add_node(self.node_counter, type="Intermediate")
                        self.node_counter += 1
                        
                    # 添加元件
                    self._add_component_to_graph(comp_idx, final_n1, final_n2)
                    return True
        
        elif action_type == 2: # 重连元件 (Rewire)
            # 只有当元件已被放置时才能重连
            if comp_idx < self.max_components and self.available_components[comp_idx] == 0:
                # 找到该元件在图中的边
                target_u, target_v, target_key = None, None, None
                
                for u, v, key, data in self.circuit_graph.edges(keys=True, data=True):
                    if data.get('inventory_idx') == comp_idx:
                        target_u, target_v, target_key = u, v, key
                        break
                
                if target_u is not None:
                    # 检查新节点是否有效 (必须是现有节点)
                    # 重连不允许创建新节点 (简化逻辑)
                    if raw_n1 >= self.node_counter or raw_n2 >= self.node_counter:
                        return False
                    
                    if raw_n1 == raw_n2:
                        return False
                        
                    # 移除旧边
                    self.circuit_graph.remove_edge(target_u, target_v, key=target_key)
                    
                    # 添加新边
                    # 注意: 需要保留元件属性
                    comp_template = self.initial_inventory[comp_idx]
                    comp_instance = deepcopy(comp_template)
                    comp_instance.nodes = (raw_n1, raw_n2)
                    
                    self.circuit_graph.add_edge(raw_n1, raw_n2, component=comp_instance, inventory_idx=comp_idx)
                    return True
                    
        return False

    def _add_component_to_graph(self, comp_idx, n1, n2):
        """辅助函数：将元件添加到图中并更新库存"""
        comp_template = self.initial_inventory[comp_idx]
        comp_instance = deepcopy(comp_template)
        comp_instance.nodes = (n1, n2)
        
        self.circuit_graph.add_edge(n1, n2, component=comp_instance, inventory_idx=comp_idx)
        self.available_components[comp_idx] = 0

    def _calculate_reward(self, result, valid_action, action):
        """
        计算奖励函数。
        """
        # 奖励参数定义
        R_SN = -1.0   # 小惩罚: 未形成新环路
        R_MN = -10.0  # 中惩罚: 形成无效环路 / 严重错误
        R_SP = +2.0   # 小奖励: 形成有效新环路
        R_MP = +10.0  # 中奖励: 所有电源都在环路中
        R_BP = +150.0 # 大奖励: 功能验证通过 (伏秒平衡)
        
        action_type = action[0]

        # --- 1. 过程奖励 (Step Rewards) ---
        
        if not valid_action:
            return -1.0 # 无效动作给予惩罚
            
        reward = 0.0
        
        if action_type == 1: # 放置
            reward = 1.0 # 基础奖励
        elif action_type == 2: # 重连
            reward = -0.5 # 重连有轻微惩罚，鼓励一次做对，但允许修改
            
        # --- 过程中的启发式奖励 (面包屑) ---
        # 鼓励将电源和开关连接在一起 (Buck 的关键第一步)
        current_comp_idx = action[1]
        if current_comp_idx < self.max_components:
            comp_template = self.initial_inventory[current_comp_idx]
            if isinstance(comp_template, (Switch, VoltageSource)):
                for u, v, data in self.circuit_graph.edges(data=True):
                    if data.get('inventory_idx') == current_comp_idx:
                        target_type = VoltageSource if isinstance(comp_template, Switch) else Switch
                        for _, _, other_data in self.circuit_graph.edges(u, data=True):
                            if isinstance(other_data['component'], target_type):
                                reward += 2.0 
                                break
                        for _, _, other_data in self.circuit_graph.edges(v, data=True):
                            if isinstance(other_data['component'], target_type):
                                reward += 2.0 
                                break
                        break
        
        # 检查是否形成了新环路
        try:
            current_cycles = nx.cycle_basis(self.circuit_graph.to_undirected())
            num_cycles = len(current_cycles)
        except:
            num_cycles = 0
            current_cycles = []
            
        if num_cycles > self.last_cycle_count:
            # 发现了新环路
            has_bad_loop = False
            for cycle_nodes in current_cycles:
                loop_components = []
                for i in range(len(cycle_nodes)):
                    u = cycle_nodes[i]
                    v = cycle_nodes[(i + 1) % len(cycle_nodes)]
                    if self.circuit_graph.has_edge(u, v):
                        edges = self.circuit_graph.get_edge_data(u, v)
                        for key, data in edges.items():
                            loop_components.append(data['component'])
                            break 
                has_inductor = any(isinstance(c, Inductor) for c in loop_components)
                has_switch_or_diode = any(isinstance(c, (Switch, Diode)) for c in loop_components)
                if not has_inductor: has_bad_loop = True
                if not has_switch_or_diode: has_bad_loop = True
            
            if has_bad_loop:
                reward -= 50.0 
            else:
                reward += R_SP 
        else:
            if action_type == 1: # 只有放置时才惩罚未闭合，重连不惩罚
                reward += R_SN 
            
        self.last_cycle_count = num_cycles
        
        # --- 2. 终局奖励 (Terminal Rewards) ---
        # 只有在所有元件都放置完毕后才计算终局奖励
        # 即使在重连阶段，每次重连后也计算一次"准终局"奖励，告诉它改得对不对
        
        if np.sum(self.available_components) == 0:
            
            # 检查 1: 所有电压源是否都在环路中
            all_sources_in_loops = True
            sources = [d['component'] for u, v, d in self.circuit_graph.edges(data=True) if isinstance(d['component'], VoltageSource)]
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
            
            # 检查 2: 桥接边检查 (Bridge Check)
            bridges = list(nx.bridges(self.circuit_graph.to_undirected()))
            if len(bridges) > 0:
                reward -= 50.0 
                # 注意：在重连模式下，我们不立即结束，而是给负分让它改
                # return reward 
            
            # --- 启发式奖励 ---
            for src in sources:
                u, v = src.nodes
                edges = self.circuit_graph.get_edge_data(u, v)
                if len(edges) > 1: 
                    reward -= 20.0 
            
            for node in self.circuit_graph.nodes():
                if self.circuit_graph.degree(node) == 2:
                    edges = list(self.circuit_graph.edges(node, data=True))
                    comp1 = edges[0][2]['component']
                    comp2 = edges[1][2]['component']
                    types = [type(comp1), type(comp2)]
                    if Inductor in types and Switch in types:
                        reward += 5.0 
                        
            # 检查 3: 功能性验证
            from utils.mode_analysis import analyze_switching_modes
            if not nx.is_connected(self.circuit_graph.to_undirected()):
                reward += R_MN 
            else:
                modes = analyze_switching_modes(self.circuit_graph)
                has_short = False
                has_increasing = False
                has_decreasing = False
                for state, data in modes.items():
                    if not data['valid']: has_short = True
                    if data['valid']:
                        trends = data['inductor_trends']
                        for comp_name, trend in trends.items():
                            if "Increasing" in trend: has_increasing = True
                            if "Decreasing" in trend: has_decreasing = True
                
                if has_short:
                    reward += R_MN 
                elif has_increasing and has_decreasing:
                    reward += R_BP 
                else:
                    reward += R_MN
                
        return reward

    def _get_obs(self):
        """
        生成当前的观测向量。
        """
        # 1. 邻接矩阵 (Adjacency Matrix)
        # 现在存储元件类型 ID (0-8)
        adj = np.zeros((self.max_nodes, self.max_nodes), dtype=np.int8)
        
        for u, v, data in self.circuit_graph.edges(data=True):
            if u < self.max_nodes and v < self.max_nodes:
                comp = data.get('component')
                if comp:
                    type_id = comp.get_type_id()
                    adj[u, v] = type_id
                    adj[v, u] = type_id
        
        # 2. 节点特征 (Node Features)
        node_feats = np.zeros((self.max_nodes, 2), dtype=np.float32)
        
        for node in range(self.node_counter): # 只更新已存在的节点
            degree = self.circuit_graph.degree(node)
            node_feats[node, 0] = float(degree)
            node_feats[node, 1] = 1.0 if degree > 0 else 0.0
            
        return {
            "adjacency": adj,
            "inventory_mask": self.available_components.copy(),
            "node_features": node_feats,
        }

    def action_masks(self) -> List[bool]:
        """
        返回动作掩码 (Action Mask)。
        """
        masks = []
        
        # 1. 动作类型掩码 (维度=3)
        # 0=增加节点 (禁用)
        # 1=放置元件 (仅当有剩余元件时可用)
        # 2=重连元件 (仅当无剩余元件时可用 -> 进入重连阶段)
        
        has_available = np.sum(self.available_components) > 0
        
        if has_available:
            masks.append([False, True, False]) # 只能放置
        else:
            masks.append([False, False, True]) # 只能重连
        
        # 2. 元件索引掩码 (维度=max_components)
        if has_available:
            # 放置模式：只能选未用的
            comp_mask = [bool(x) for x in self.available_components]
        else:
            # 重连模式：只能选已用的 (即所有元件)
            comp_mask = [True] * self.max_components
            
        masks.append(comp_mask)
        
        # 3. 节点1 掩码 (维度=max_nodes)
        node_mask = [False] * self.max_nodes
        
        if has_available:
            # 放置模式逻辑 (同前)
            if self.node_counter == 0:
                node_mask[0] = True
                if self.max_nodes > 1: node_mask[1] = True
            else:
                for i in range(self.node_counter):
                    node_mask[i] = True
                if self.node_counter < self.max_nodes:
                    node_mask[self.node_counter] = True
        else:
            # 重连模式逻辑
            # 只能连接现有节点
            for i in range(self.node_counter):
                node_mask[i] = True
                
        masks.append(node_mask)
        
        # 4. 节点2 掩码 (维度=max_nodes)
        masks.append(node_mask.copy()) 
        
        flattened_mask = []
        for m in masks:
            flattened_mask.extend(m)
            
        return flattened_mask

    def render(self):
        pass


