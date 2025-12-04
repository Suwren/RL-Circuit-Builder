import networkx as nx
import itertools
from copy import deepcopy
import numpy as np
from env.simulator import CircuitSimulator
from env.components import Switch, VoltageSource, Inductor, Diode, Wire, Capacitor

def analyze_switching_modes(graph: nx.MultiGraph):
    """
    分析电路在所有可能的开关状态组合下的模态。
    
    参数:
        graph: 电路的 NetworkX MultiGraph 表示。
        
    返回:
        results: 字典，键为开关状态元组，值为该模态的分析结果。
    """
    # 1. 识别电路中的所有开关
    switches = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        comp = data.get('component')
        if isinstance(comp, Switch):
            switches.append({'edge': (u, v, key), 'name': comp.name, 'obj': comp})
            
    # 按名称排序以保证状态顺序一致
    switches.sort(key=lambda x: x['name'])
    switch_names = [s['name'] for s in switches]
    
    results = {}
    num_switches = len(switches)
    # 生成所有开关状态组合 (True=闭合, False=断开)
    state_combinations = list(itertools.product([False, True], repeat=num_switches))
    
    for states in state_combinations:
        mode_graph = graph.copy()
        
        # 应用当前组合的开关状态
        for i, state in enumerate(states):
            sw_info = switches[i]
            u, v, key = sw_info['edge']
            
            if mode_graph.has_edge(u, v, key):
                edge_data = mode_graph.get_edge_data(u, v, key)
                # 深拷贝元件并更新状态
                new_comp = deepcopy(edge_data['component'])
                new_comp.state = state 
                edge_data['component'] = new_comp
        
        # 分析当前模态的特性
        analysis = analyze_mode_characteristics(mode_graph)
        
        results[states] = {
            'switches': switch_names,
            'graph': mode_graph,
            'valid': analysis['valid'],
            'reasons': analysis['reasons'],
            'inductor_trends': analysis['inductor_trends']
        }
        
    return results

def analyze_mode_characteristics(graph: nx.MultiGraph):
    """
    分析特定开关状态下的电路特性。
    包括：
    1. 拓扑连通性检查 (电感续流路径)。
    2. SPICE 仿真 (短路检测和电感电流趋势)。
    """
    valid = True
    reasons = []
    inductor_trends = {}
    
    # --- 1. 拓扑连通性检查 (基于图搜索) ---
    # 目的：检查每个电感是否拥有合法的续流回路，而不依赖 SPICE 仿真。
    
    # 构建流图 (Flow Graph) - 有向图
    # 节点: 电路节点
    # 边: 允许电流流过的路径
    flow_graph = nx.DiGraph()
    flow_graph.add_nodes_from(graph.nodes())
    
    for u, v, key, data in graph.edges(keys=True, data=True):
        comp = data.get('component')
        if not comp: continue
        
        # 根据元件类型添加有向边
        if isinstance(comp, (VoltageSource, Wire, Capacitor)):
            # 双向导通
            flow_graph.add_edge(u, v)
            flow_graph.add_edge(v, u)
            
        elif isinstance(comp, Switch):
            if comp.state: # 开关闭合
                # 双向导通
                flow_graph.add_edge(u, v)
                flow_graph.add_edge(v, u)
            else: # 开关断开
                # 只有体二极管导通
                # 假设 n1=Drain, n2=Source. Body Diode: Source(n2) -> Drain(n1)
                n1, n2 = comp.nodes
                # 确保 u, v 匹配 n1, n2 的方向
                if u == n1 and v == n2:
                    flow_graph.add_edge(v, u) # n2 -> n1
                else:
                    flow_graph.add_edge(u, v) # n2 -> n1 (u is n2)
                    
        elif isinstance(comp, Diode):
            # 二极管单向导通: Anode -> Cathode
            n1, n2 = comp.nodes # n1=Anode, n2=Cathode
            if u == n1 and v == n2:
                flow_graph.add_edge(u, v)
            else:
                flow_graph.add_edge(v, u)
                
        elif isinstance(comp, Inductor):
            # 电感本身是我们要检查的对象，不作为外部回路的一部分
            pass

    # 检查每个电感是否有外部回路
    inductors = []
    for u, v, data in graph.edges(data=True):
        if isinstance(data['component'], Inductor):
            inductors.append({'nodes': (u, v), 'obj': data['component']})
            
    for item in inductors:
        ind = item['obj']
        n1, n2 = ind.nodes
        
        # 检查方向 1: 电流 n1 -> n2
        # 需要外部回路 n2 -> ... -> n1
        has_path_fwd = nx.has_path(flow_graph, n2, n1)
        
        # 检查方向 2: 电流 n2 -> n1
        # 需要外部回路 n1 -> ... -> n2
        has_path_rev = nx.has_path(flow_graph, n1, n2)
        
        # 只要有一个方向通，就认为电感没有完全断路
        if not (has_path_fwd or has_path_rev):
            valid = False
            reasons.append(f"Inductor {ind.name} Open Circuit (No path for current)")
            
    # --- 2. 运行 SPICE 仿真 ---
    # 仅在拓扑检查通过后运行，用于检测短路和分析趋势
    
    simulator = CircuitSimulator()
    
    if valid:
        # 运行快照仿真 (Snapshot/Transient)
        analysis = simulator.run_snapshot(graph) 
        
        if analysis is None:
            return {
                'valid': False,
                'reasons': ["Simulation Failed (Singular Matrix / Invalid Topology)"],
                'inductor_trends': {}
            }
            
        # --- 3. 检查短路 (已移除) ---
        # 我们现在完全依赖拓扑检查 (Loop must have Inductor) 来防止短路。
        # SPICE 仿真仅用于趋势分析。
        pass
    else:
        # 如果拓扑已经无效，跳过仿真
        analysis = None
                
    # --- 4. 分析电感电流趋势 ---
    # 只有在电路有效时才分析
    if valid:
        inductors = []
        for u, v, data in graph.edges(data=True):
            if isinstance(data['component'], Inductor):
                inductors.append({'nodes': (u, v), 'obj': data['component']})
                
        for item in inductors:
            ind = item['obj']
            n1_name = str(ind.nodes[0])
            n2_name = str(ind.nodes[1])
            
            try:
                # 获取电感两端电压
                v1 = float(analysis.nodes[n1_name][0]) if n1_name in analysis.nodes else 0.0
                v2 = float(analysis.nodes[n2_name][0]) if n2_name in analysis.nodes else 0.0
                
                v_ind = v1 - v2
                
                # 根据 V = L * di/dt 判断趋势
                # V > 0 -> di/dt > 0 (Increasing)
                # V < 0 -> di/dt < 0 (Decreasing)
                
                if v_ind > 1e-3: # 正电压阈值
                    trend = "Increasing (di/dt > 0)"
                elif v_ind < -1e-3: # 负电压阈值
                    trend = "Decreasing (di/dt < 0)"
                else:
                    trend = "Constant / Zero"
                    
                inductor_trends[ind.name] = trend
                
            except KeyError:
                inductor_trends[ind.name] = "Unknown (Node not found)"
 
    return {
        'valid': valid,
        'reasons': reasons,
        'inductor_trends': inductor_trends
    }
