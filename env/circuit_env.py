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
        # 动作类型: 0=增加节点 (已禁用/保留兼容), 1=放置元件
        self.action_space = spaces.MultiDiscrete([2, self.max_components, max_nodes, max_nodes])

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
                    
                    # 解释节点选择:
                    # 如果 ID < current_node_count -> 现有节点
                    # 如果 ID >= current_node_count -> 请求新节点
                    
                    # 确定实际的节点 ID
                    # 如果请求新节点，我们暂时分配一个新的 ID (current_node_count)
                    # 注意：如果两个都请求新节点，它们是同一个新节点吗？
                    # 简化逻辑：如果 raw_n >= current，则视为"连接到一个新创建的节点"
                    # 如果 raw_n1 和 raw_n2 都 >= current，则意味着连接两个新节点 -> 这在"后续元件"逻辑中是不允许的(孤岛)
                    
                    is_n1_new = raw_n1 >= current_node_count
                    is_n2_new = raw_n2 >= current_node_count
                    
                    # 约束检查: 必须至少有一个连接到现有节点
                    if is_n1_new and is_n2_new:
                        return False # 试图创建孤岛
                    
                    # 确定最终的节点 ID
                    final_n1 = raw_n1 if not is_n1_new else current_node_count
                    # 如果 n1 是新的，n2 必须是旧的。如果 n2 也是新的(上面已拦截)，或者 n2 是旧的。
                    # 如果 n1 是旧的，n2 可以是新的。如果 n2 是新的，它的 ID 应该是 current_node_count。
                    # 此时如果 n1 也是新的(不可能，已拦截)。
                    
                    # 这里的逻辑有点微妙：如果 n1 和 n2 都是"新"的，我们拒绝。
                    # 如果只有一个是新的，那个新的 ID 就是 current_node_count。
                    # 如果两个都是旧的，直接用。
                    
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
        基于论文策略：过程奖励 (环路形成) + 终局奖励 (功能验证)。
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
            
        reward = 1.0 # 基础奖励: 鼓励有效连接 (原为 0.5)
        
        # --- 过程中的启发式奖励 (面包屑) ---
        # 鼓励将电源和开关连接在一起 (Buck 的关键第一步)
        # 检查刚添加的边是否连接了电源和开关
        # action: [type, comp_idx, n1, n2]
        # 注意: 这里我们只检查刚刚放置的元件
        # 如果放置的是开关，检查 n1, n2 是否连接了电源
        # 如果放置的是电源，检查 n1, n2 是否连接了开关
        
        # 获取刚刚放置的元件实例
        # 注意: _apply_action 已经将元件加入图中
        # 我们需要获取刚刚操作的节点 n1, n2 (经过 _apply_action 处理后的实际节点)
        # 由于 _apply_action 没有返回实际节点，我们需要从图中推断，或者修改 _apply_action 返回值。
        # 简化方案：遍历该元件的所有连接。
        
        # 获取当前放置的元件类型
        current_comp_idx = action[1]
        if current_comp_idx < self.max_components:
            # 注意: self.initial_inventory 是模板，我们需要检查图中实际连接的情况
            # 但我们可以根据 comp_idx 知道是什么类型的元件
            comp_template = self.initial_inventory[current_comp_idx]
            
            # 只有当放置的是开关或电源时才检查
            if isinstance(comp_template, (Switch, VoltageSource)):
                # 找到这个元件在图中的实例
                # 遍历图中的边，找到 inventory_idx == current_comp_idx 的边
                for u, v, data in self.circuit_graph.edges(data=True):
                    if data.get('inventory_idx') == current_comp_idx:
                        # 检查 u 或 v 是否连接了另一类元件
                        target_type = VoltageSource if isinstance(comp_template, Switch) else Switch
                        
                        # 检查节点 u 的其他连接
                        for _, _, other_data in self.circuit_graph.edges(u, data=True):
                            if isinstance(other_data['component'], target_type):
                                reward += 2.0 # 奖励电源-开关连接
                                break
                                
                        # 检查节点 v 的其他连接
                        for _, _, other_data in self.circuit_graph.edges(v, data=True):
                            if isinstance(other_data['component'], target_type):
                                reward += 2.0 # 奖励电源-开关连接
                                break
                        break
        
        # 检查是否形成了新环路
        try:
            # 获取当前的环路基 (Cycle Basis)
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
                reward -= 50.0 # 严厉惩罚不可控回路 (原为 -10.0)
            else:
                reward += R_SP # 发现好环路，奖励
                
        else:
            # 没有发现新环路
            reward += R_SN # 鼓励尽快闭合回路
            
        self.last_cycle_count = num_cycles
        
        # 额外检查: 并联开关惩罚
        # 智能体不应在两点间并联多个开关
        # 注意：由于现在节点是动态生成的，并联开关的可能性变小了，但仍然存在
        n1, n2 = 0, 0 # 这里需要获取实际连接的节点，但 _apply_action 内部处理了。
        # 我们可以通过检查图来获取最近添加的边，或者简化这个检查。
        # 简单起见，我们遍历全图检查是否有并联开关 (性能稍差但准确)
        # 或者，由于我们知道动作意图，我们可以尝试推断。
        # 这里为了准确性，我们暂时跳过针对"本步"的特定检查，而是依赖全局检查或后续优化。
        # 实际上，如果两个节点间有多个开关，这在 cycle check 中可能不会直接体现，但在功能分析中会体现。
        # 我们可以保留一个简单的检查：
        for u, v, data in self.circuit_graph.edges(data=True):
            if isinstance(data['component'], Switch):
                # 检查该边是否有其他开关并联
                edges = self.circuit_graph.get_edge_data(u, v)
                switch_count = sum(1 for k, d in edges.items() if isinstance(d['component'], Switch))
                if switch_count > 1:
                    reward += R_MN
                    break
        
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
            
            # 检查 2: 桥接边检查 (Bridge Check)
            bridges = list(nx.bridges(self.circuit_graph.to_undirected()))
            if len(bridges) > 0:
                reward -= 50.0 
                return reward 
            
            # --- 启发式奖励 (Heuristic Rewards) ---
            # 目的: 引导智能体跳出局部最优 (如 Vin || L || S 短路)
            
            # 1. 短路惩罚 (Short Circuit Penalty)
            # 检查电压源是否在长度为 2 的环路中 (即直接并联)
            # Vin || L -> 短路
            # Vin || S -> 短路 (当 S 闭合时)
            # Vin || Wire -> 短路
            for src in sources:
                u, v = src.nodes
                # 检查 u, v 之间是否有其他边
                edges = self.circuit_graph.get_edge_data(u, v)
                if len(edges) > 1: # 超过 1 条边，说明有并联
                    reward -= 20.0 # 惩罚电源直接并联
            
            # 2. 串联连接奖励 (Series Connection Reward)
            # 鼓励电感和开关串联 (Buck 的特征)
            # 判据: 某节点的度数为 2，且连接了一个电感和一个开关
            for node in self.circuit_graph.nodes():
                if self.circuit_graph.degree(node) == 2:
                    # 获取连接该节点的两条边
                    edges = list(self.circuit_graph.edges(node, data=True))
                    comp1 = edges[0][2]['component']
                    comp2 = edges[1][2]['component']
                    
                    types = [type(comp1), type(comp2)]
                    if Inductor in types and Switch in types:
                        reward += 5.0 # 奖励电感-开关串联
                        
            # 检查 3: 功能性验证 (伏秒平衡)

            # 检查 3: 功能性验证 (伏秒平衡)
            from utils.mode_analysis import analyze_switching_modes
            
            # 连通性检查 (现在由构建过程保证，但为了保险还是留着)
            if not nx.is_connected(self.circuit_graph.to_undirected()):
                reward += R_MN 
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
        # Feature 0: 节点度数 (Degree)
        # Feature 1: 是否已连接 (Binary, Degree > 0)
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
        True 表示动作有效，False 表示动作无效。
        MaskablePPO 会使用此掩码将无效动作的概率置为 0。
        
        动作空间: MultiDiscrete([2, max_components, max_nodes, max_nodes])
        展平后的维度: 2 + max_components + max_nodes + max_nodes
        注意: MaskablePPO 期望的掩码是针对展平后的动作空间的吗？
        不，对于 MultiDiscrete，sb3-contrib 期望返回一个列表，其中每个元素对应一个维度的掩码。
        或者是一个拼接的大数组？
        
        查阅文档/源码: sb3_contrib 的 MaskablePPO 对 MultiDiscrete 的支持比较特殊。
        通常它期望一个展平的掩码，或者针对每个维度的掩码列表。
        但在 MultiDiscrete 情况下，MaskablePPO 目前可能只支持 Discrete 空间，或者对 MultiDiscrete 的支持有限。
        
        修正: sb3-contrib 的 MaskablePPO 确实支持 MultiDiscrete，但掩码必须是一个列表，
        列表中的每个元素是一个布尔数组，对应 MultiDiscrete 的一个维度。
        """
        masks = []
        
        # 1. 动作类型掩码 (维度=2)
        # 0=增加节点 (禁用), 1=放置元件 (启用)
        masks.append([False, True])
        
        # 2. 元件索引掩码 (维度=max_components)
        # 只有库存中存在的元件才可用
        # self.available_components: 1=可用, 0=已用
        comp_mask = [bool(x) for x in self.available_components]
        masks.append(comp_mask)
        
        # 3. 节点1 掩码 (维度=max_nodes)
        # 只能连接已存在的节点 (0 到 node_counter-1)
        # 或者如果是第一个元件，允许连接 0 (虽然此时 node_counter=0，但 _apply_action 会处理)
        # 实际上，_apply_action 逻辑允许连接 "current_node_count" 来创建新节点。
        # 所以有效范围是 0 到 node_counter (包含 node_counter 用于创建新节点)
        # 但不能超过 max_nodes
        
        node_mask = [False] * self.max_nodes
        
        if self.node_counter == 0:
            # 第一个元件，强制连接 0 和 1 (在 _apply_action 中处理)
            # 这里我们只需允许 0 和 1 (或者任意，因为 _apply_action 会重写)
            # 为了配合 _apply_action 的逻辑:
            # "if self.circuit_graph.number_of_nodes() == 0: n1, n2 = 0, 1"
            # 所以只要允许任何合法的索引即可。
            node_mask[0] = True
            if self.max_nodes > 1: node_mask[1] = True
        else:
            # 允许连接现有节点 [0, node_counter-1]
            for i in range(self.node_counter):
                node_mask[i] = True
            
            # 允许连接下一个新节点 (node_counter)，前提是没超过 max_nodes
            if self.node_counter < self.max_nodes:
                node_mask[self.node_counter] = True
                
        masks.append(node_mask)
        
        # 4. 节点2 掩码 (维度=max_nodes)
        # 逻辑同节点1
        masks.append(node_mask.copy()) # 使用相同的掩码
        
        # MaskablePPO 对于 MultiDiscrete 空间，期望返回一个展平的 1D 列表
        # 即所有维度的掩码拼接在一起
        flattened_mask = []
        for m in masks:
            flattened_mask.extend(m)
            
        return flattened_mask

    def render(self):
        pass


