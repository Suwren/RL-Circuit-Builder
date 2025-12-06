import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from copy import deepcopy

from env.components import Component, Inductor, Capacitor, Resistor, VoltageSource, Switch, Diode, Wire
from utils.mode_analysis import analyze_switching_modes, check_static_safety

class CircuitEnv(gym.Env):
    """
    Gymnasium Environment for Power Electronic Circuit Construction.
    
    The agent builds a circuit by placing components from an inventory onto a breadboard-like grid.
    The goal is to construct a functional converter (e.g., Buck Converter) that satisfies
    volt-second balance and has no short circuits.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, initial_components: List[Component], max_nodes: int = 20, verbose: bool = False):
        super().__init__()
        self.max_nodes = max_nodes
        self.initial_inventory = deepcopy(initial_components)
        self.max_components = len(initial_components)
        self.verbose = verbose
        
        self.circuit_graph = nx.MultiGraph()
        self.node_counter = 0 
        
        # Action Space: [Action Type, Component ID, Node 1, Node 2]
        # Action Type: 0=Stop, 1=Place Component
        self.action_space = spaces.MultiDiscrete([2, self.max_components, max_nodes, max_nodes])

        # Observation Space
        self.observation_space = spaces.Dict({
            "adjacency": spaces.Box(low=0, high=9, shape=(max_nodes, max_nodes), dtype=np.int8),
            # Optimized Inventory: [Count_V, Count_L, Count_S]
            # Max possible count is max_components
            "inventory_counts": spaces.Box(low=0, high=self.max_components, shape=(3,), dtype=np.float32),
            "node_features": spaces.Box(low=-np.inf, high=np.inf, shape=(max_nodes, 2), dtype=np.float32),
        })

        self.reset()

    def reset(self, seed=None, options=None):
        """Resets the environment to the initial blank state."""
        super().reset(seed=seed)
        self.circuit_graph.clear()
        
        self.node_counter = 0
            
        self.step_count = 0
        self.terminated = False
        self.last_cycle_count = 0
        
        # Set max steps based on component count
        self.max_steps = self.max_components * 3 
        if self.max_steps < 10: self.max_steps = 10
        
        # Reset inventory (1 = available)
        self.available_components = np.ones(self.max_components, dtype=np.int8)
        
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        """Executes one step in the environment."""
        self.step_count += 1
        
        # 0. Handle Stop Action
        if action[0] == 0:
            self.terminated = True
            terminated = True
            reward = self._calculate_circuit_score()
            observation = self._get_obs()
            return observation, reward, terminated, False, {"valid_action": True, "simulation_result": None}

        # 1. Apply Action (Place Component)
        valid_action = self._apply_action(action)
        
        simulation_result = None
        reward = 0.0
        
        # 2. Check Termination Conditions
        terminated = False
        if np.sum(self.available_components) == 0:
            terminated = True
            self.terminated = True
            
        truncated = False
        if self.step_count >= self.max_steps:
            truncated = True
            if np.sum(self.available_components) > 0:
                reward = -10.0 # Penalty for running out of time
        
        # 3. Calculate Final Reward if Terminated
        if terminated:
            reward = self._calculate_circuit_score()

        observation = self._get_obs()
        info = {
            "valid_action": valid_action,
            "simulation_result": simulation_result
        }
        
        return observation, reward, terminated, truncated, info

    def _apply_action(self, action) -> bool:
        """
        Applies the placement action.
        Action format: [ActionType, ComponentType, Node1, Node2]
        ActionType: 0=Stop, 1=Place
        ComponentType: 0=VoltageSource, 1=Inductor, 2=Switch
        """
        action_type = action[0]
        type_idx = action[1]
        raw_n1 = action[2]
        raw_n2 = action[3]
        
        if action_type == 0:
            return False
                
        elif action_type == 1:
            # Find the first available component of the requested type
            target_class = None
            if type_idx == 0: target_class = VoltageSource
            elif type_idx == 1: target_class = Inductor
            elif type_idx == 2: target_class = Switch
            else: return False
            
            comp_idx = -1
            for i, comp in enumerate(self.initial_inventory):
                if isinstance(comp, target_class) and self.available_components[i] == 1:
                    comp_idx = i
                    break
            
            # If no component of this type is available, action is invalid
            if comp_idx == -1:
                return False
                
            # Case 1: First component
            if self.circuit_graph.number_of_nodes() == 0:
                # Respect the direction of the action for polarity
                # If model asks for 1->0 (raw_n1 > raw_n2), place at 1->0
                # If model asks for 0->1 (raw_n1 < raw_n2), place at 0->1
                # We still use physical nodes 0 and 1 to initialize the graph.
                if raw_n1 > raw_n2:
                    n1, n2 = 1, 0
                else:
                    n1, n2 = 0, 1
                    
                self.circuit_graph.add_node(0, type="Intermediate")
                self.circuit_graph.add_node(1, type="Intermediate")
                self.node_counter = 2
                
                self._add_component_to_graph(comp_idx, n1, n2)
                return True
            
            # Case 2: Subsequent components
            else:
                current_node_count = self.node_counter
                
                # Check if nodes are "new" (requested ID >= current count)
                is_n1_new = raw_n1 >= current_node_count
                is_n2_new = raw_n2 >= current_node_count
                
                # Cannot create two new nodes at once (would create an island)
                if is_n1_new and is_n2_new:
                    return False
                
                final_n1 = raw_n1 if not is_n1_new else current_node_count
                
                if is_n2_new:
                    final_n2 = current_node_count
                else:
                    final_n2 = raw_n2
                    
                # Cannot connect a node to itself
                if final_n1 == final_n2:
                    return False
                
                # Create new node if needed
                if is_n1_new or is_n2_new:
                    if self.node_counter >= self.max_nodes:
                        return False
                    
                    self.circuit_graph.add_node(self.node_counter, type="Intermediate")
                    self.node_counter += 1
                    
                self._add_component_to_graph(comp_idx, final_n1, final_n2)
                return True
        
        return False

    def _add_component_to_graph(self, comp_idx, n1, n2):
        """Helper to add component to graph and update inventory."""
        comp_template = self.initial_inventory[comp_idx]
        comp_instance = deepcopy(comp_template)
        comp_instance.nodes = (n1, n2)
        
        self.circuit_graph.add_edge(n1, n2, component=comp_instance, inventory_idx=comp_idx)
        self.available_components[comp_idx] = 0

    def _calculate_circuit_score(self):
        """
        Calculates the circuit score based on versatility across 6 Power Flow Scenarios.
        Scenarios are permutations of Input/Output roles for V1, V2, V3.
        """
        score = 0.0
        
        # 1. Strict Component Check (User Requirement)
        # Must have V1, V2, V3, and L1.
        required_components = ["V1", "V2", "V3", "L1"]
        present_components = set()
        
        for u, v, data in self.circuit_graph.edges(data=True):
            comp = data.get('component')
            if comp and comp.name in required_components:
                present_components.add(comp.name)
                
        missing_components = [name for name in required_components if name not in present_components]
        
        if missing_components:
            if self.verbose:
                print(f"  [结果] 缺少关键组件 {missing_components}。惩罚: -200.0")
            return -200.0
        
        # 2. Strict Switch Check (User Requirement)
        used_switches = len([d['component'] for u, v, d in self.circuit_graph.edges(data=True) if isinstance(d.get('component'), Switch)])
        if used_switches == 0:
            if self.verbose:
                print(f"  [结果] 未放置任何开关。惩罚: -200.0")
            return -200.0
            
        # 3. Strict Dangling Node Check (User Requirement)
        dangling_nodes = [n for n, d in self.circuit_graph.degree() if d == 1]
        if len(dangling_nodes) > 0:
            if self.verbose:
                print(f"  [结果] 存在悬空节点 {dangling_nodes}。惩罚: -100.0")
            return -100.0

        # 4. Structural Penalties (Optimization)
        # Switch Penalty: Encourage fewer switches (Linear penalty)
        switch_penalty = used_switches * 5.0
        score -= switch_penalty
        
        # Parallel Switch Penalty: discourage placing multiple switches between same nodes
        switch_locations = {}
        for u, v, d in self.circuit_graph.edges(data=True):
            if isinstance(d.get('component'), Switch):
                # Use sorted tuple to identify the pair regardless of direction
                pair = tuple(sorted((u, v)))
                switch_locations[pair] = switch_locations.get(pair, 0) + 1
        
        parallel_penalty = 0.0
        for count in switch_locations.values():
            if count > 1:
                # Deduct 10 points for each redundant switch
                parallel_penalty += (count - 1) * 10.0
        
        if parallel_penalty > 0:
            if self.verbose:
                print(f"  [结构惩罚] 存在并联开关。惩罚: -{parallel_penalty}")
            score -= parallel_penalty
        
        # Basic connectivity checks
        if self.circuit_graph.number_of_nodes() == 0:
            return score
            
        if not nx.is_connected(self.circuit_graph.to_undirected()):
            if self.verbose:
                print(f"  [结果] 电路不连通 (Disconnected Graph)。惩罚: -100.0")
            return -100.0
            
        from utils.mode_analysis import analyze_switching_modes
        
        # 定义电压源的角色分配场景 (Define the 3 Scenarios)
        # 每个元组格式: (V1_role, V2_role, V3_role)
        # 根据用户要求，V1 固定为 "output"。
        # We only consider 3 scenarios where V1 is always an OUTPUT.
        scenarios = [
            ("output", "input", "output"), # 场景 1: V2 输入, V1/V3 输出 (Dual Output)
            ("output", "output", "input"), # 场景 2: V3 输入, V1/V2 输出 (Dual Output)
            ("output", "input", "input"),  # 场景 3: V2/V3 输入, V1 输出 (Dual Input)
        ]
        
        total_versatility_score = 0.0
        
        # Pre-fetch component instances for efficiency
        v_components = {}
        for u, v, key, data in self.circuit_graph.edges(keys=True, data=True):
            comp = data.get('component')
            if isinstance(comp, VoltageSource) and comp.name in ["V1", "V2", "V3"]:
                v_components[comp.name] = comp
        
        for i, (r1, r2, r3) in enumerate(scenarios):
            scenario_name = f"Scenario {i+1} (V1:{r1}, V2:{r2}, V3:{r3})"
            if self.verbose: print(f"\nEvaluating {scenario_name}...")
            
            # Configure Roles
            v_components["V1"].role = r1
            v_components["V2"].role = r2
            v_components["V3"].role = r3
            
            inputs = [name for name, role in [("V1", r1), ("V2", r2), ("V3", r3)] if role == "input"]
            outputs = [name for name, role in [("V1", r1), ("V2", r2), ("V3", r3)] if role == "output"]
            
            scenario_score = 0.0
            is_static_safe = True
            
            # 3. Static Safety Check
            test_graph = self.circuit_graph.copy()
            
            for inp in inputs:
                for out in outputs:
                    is_safe, msg = check_static_safety(test_graph, inp, out)
                    if not is_safe:
                        is_static_safe = False
                        if self.verbose: print(f"  [静态检查失败] {inp} -> {out}: {msg}")
                        break
                if not is_static_safe: break
            
            if not is_static_safe:
                if self.verbose: print(f"  [结果] {scenario_name}: 静态安全检查未通过。得分: 0.0")
                continue 
                
            # Static check passed
            scenario_score += 10.0 
            if self.verbose: print(f"  [得分详情] 静态安全检查通过: +10.0")
            
            # 4. Dynamic Functional Check (Inductor-Based)
            # Check if Inductor L1 current can be regulated (Increase AND Decrease)
            # We pick one output to guide the mode analysis (e.g., the first output)
            # Ideally, mode analysis should be independent of load, but the function requires a load_name
            # to determine the path. We use the first output as the reference load.
            ref_load = outputs[0]
            
            modes = analyze_switching_modes(test_graph, ref_load)
            
            valid_modes_count = 0
            has_increasing = False
            has_decreasing = False
            
            for state, data in modes.items():
                # Construct switch state string for logging
                switches = data.get('switches', [])
                switch_state_str = ", ".join([f"{sw}={'ON' if s else 'OFF'}" for sw, s in zip(switches, state)])
                
                if data.get('is_shorted'):
                    reasons = ", ".join(data.get('reasons', []))
                    if self.verbose: print(f"  [模态分析] 开关状态 [{switch_state_str}]: 短路 (已忽略) - 原因: {reasons}")
                    
                    # [NEW] Strict Single-Switch Short Check
                    # If shorted AND only 1 switch is ON -> Return -100.0 immediately
                    if sum(state) == 1:
                        if self.verbose: print(f"  [严重错误] 单开关导通导致短路！本场景得分归零。")
                        scenario_score = 0.0
                        break
                    
                elif data.get('valid'):
                    valid_modes_count += 1
                    trends = data.get('inductor_trends', {})
                    l1_trend = trends.get("L1", set())
                    
                    trends_str = ", ".join([f"{k}: {v}" for k, v in trends.items()])
                    if self.verbose: print(f"  [模态分析] 开关状态 [{switch_state_str}]: 有效。电感趋势: {{{trends_str}}}")
                    
                    if "Increasing" in l1_trend: has_increasing = True
                    if "Decreasing" in l1_trend: has_decreasing = True
                else:
                    reasons = ", ".join(data.get('reasons', []))
                    if self.verbose: print(f"  [模态分析] 开关状态 [{switch_state_str}]: 无效。原因: {reasons}")
            
            if has_increasing and has_decreasing:
                scenario_score += 50.0
                if self.verbose: print(f"  [功能检查通过] 电感 L1 具备完整调节能力 (增+减): +50.0")
            else:
                if self.verbose: print(f"  [功能检查失败] 电感 L1 无法完全调节 (增:{has_increasing}, 减:{has_decreasing})")
                # Partial credit (+10) is already in scenario_score from static check
                pass
                
            if self.verbose: print(f"  [场景总分] {scenario_name}: {scenario_score}")
            total_versatility_score += scenario_score
            
        score += total_versatility_score
        
        if self.verbose:
            print(f"\n--- 最终评分计算 ---")
            print(f"  功能性得分: {total_versatility_score}")
            print(f"  开关数量惩罚: -{switch_penalty}")
            if parallel_penalty > 0:
                print(f"  并联开关惩罚: -{parallel_penalty}")
            print(f"  总分: {score}")
            
        return score

    def _get_obs(self):
        """Generates the observation dictionary."""
        adj = np.zeros((self.max_nodes, self.max_nodes), dtype=np.int8)
        
        # Sort edges to ensure deterministic observation
        edges = list(self.circuit_graph.edges(data=True))
        edges.sort(key=lambda x: (x[0], x[1], x[2]['component'].name if x[2].get('component') else ""))
        
        for u, v, data in edges:
            if u < self.max_nodes and v < self.max_nodes:
                comp = data.get('component')
                if comp:
                    type_id = comp.get_type_id() # 1=V, 2=L, 3=S, 0=Others
                    
                    if type_id > 0:
                        # Directional Encoding:
                        # adj[n1, n2] = Forward Type (1, 2, 3)
                        # adj[n2, n1] = Reverse Type (4, 5, 6)
                        
                        n1, n2 = comp.nodes
                        
                        # Ensure we don't go out of bounds (though graph nodes should be valid)
                        if n1 < self.max_nodes and n2 < self.max_nodes:
                            adj[n1, n2] = type_id
                            adj[n2, n1] = type_id + 3
        
        node_feats = np.zeros((self.max_nodes, 2), dtype=np.float32)
        
        for node in range(self.node_counter):
            degree = self.circuit_graph.degree(node)
            node_feats[node, 0] = float(degree)
            node_feats[node, 1] = 1.0 if degree > 0 else 0.0
            
        # Inventory Counts: [Count Voltage, Count Inductor, Count Switch]
        # 用户要求：库存剩余信息矩阵中被广播的单一数值从比例改为剩余个数
        # User Requirement: Use absolute counts instead of normalized ratios
        total_v = 0
        total_l = 0
        total_s = 0
        
        for i, comp in enumerate(self.initial_inventory):
            if self.available_components[i] == 1:
                if isinstance(comp, VoltageSource): total_v += 1
                elif isinstance(comp, Inductor): total_l += 1
                elif isinstance(comp, Switch): total_s += 1
        
        inv_counts = np.array([total_v, total_l, total_s], dtype=np.float32)

        return {
            "adjacency": adj,
            "inventory_counts": inv_counts,
            "node_features": node_feats,
        }

    def action_masks(self) -> List[bool]:
        """
        Returns a list of boolean masks for each dimension of the MultiDiscrete action space.
        Used by MaskablePPO / MCTS to filter invalid actions.
        """
        masks = []
        
        # 1. Action Type (Stop/Place)
        # 用户要求：智能体至少在第 6 步才可以选择完成拓扑/停止生成
        # User Requirement: Can only choose "Stop" at step >= 6
        # self.step_count starts at 0.
        # step_count=0 (1st Action) -> Cannot Stop
        # ...
        # step_count=4 (5th Action) -> Cannot Stop
        # step_count=5 (6th Action) -> Can Stop
        can_stop = self.step_count >= 5
        masks.append([can_stop, True])
        
        # 2. Component Type Mask
        # Check availability of each type (V, L, S)
        comp_mask = [False] * 3
        
        has_voltage = any(isinstance(c, VoltageSource) and self.available_components[i] == 1 for i, c in enumerate(self.initial_inventory))
        has_inductor = any(isinstance(c, Inductor) and self.available_components[i] == 1 for i, c in enumerate(self.initial_inventory))
        has_switch = any(isinstance(c, Switch) and self.available_components[i] == 1 for i, c in enumerate(self.initial_inventory))
        
        comp_mask[0] = has_voltage
        comp_mask[1] = has_inductor
        comp_mask[2] = has_switch
        masks.append(comp_mask)
        
        # 3. Node Selection
        node_mask = [False] * self.max_nodes
        
        if self.node_counter == 0:
            # First component: allow arbitrary valid indices (will be mapped to 0, 1)
            node_mask[0] = True
            if self.max_nodes > 1: node_mask[1] = True
        else:
            # Allow existing nodes
            for i in range(self.node_counter):
                node_mask[i] = True
            
            # Allow creating ONE new node
            if self.node_counter < self.max_nodes:
                node_mask[self.node_counter] = True
                
        masks.append(node_mask)
        masks.append(node_mask.copy()) # Same mask for Node 2
        
        # Flatten for sb3-contrib compatibility
        flattened_mask = []
        for m in masks:
            flattened_mask.extend(m)
            
        return flattened_mask
