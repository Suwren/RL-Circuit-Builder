import networkx as nx
import itertools
from copy import deepcopy
import numpy as np
from env.components import Switch, VoltageSource, Inductor, Diode, Wire, Capacitor, Resistor

def _format_path_with_components(graph: nx.Graph, path: list) -> str:
    """
    格式化路径字符串，包含节点和组件名称。
    Format path string including nodes and component names.
    Example: Node 1 -> [Switch S1] -> Node 2
    """
    if not path or len(path) < 2:
        return str(path)
        
    result = []
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        result.append(f"节点{u}")
        
        # 尝试获取边上的组件
        edge_data = graph.get_edge_data(u, v)
        # MultiGraph 可能有多条边，这里简单取第一条或者根据权重/key
        # 在 flow_graph 中通常是 DiGraph，直接获取
        if edge_data:
            # 如果是 MultiGraph，edge_data 可能包含 key
            if isinstance(graph, nx.MultiGraph):
                # 取第一个
                key = list(edge_data.keys())[0]
                data = edge_data[key]
            else:
                data = edge_data
                
            comp = data.get('component')
            if comp:
                result.append(f" -> [{comp.name} ({type(comp).__name__})] -> ")
            else:
                result.append(" -> [导线/未知] -> ")
        else:
            result.append(" -> ")
            
    result.append(f"节点{path[-1]}")
    return "".join(result)

def check_static_safety(graph: nx.MultiGraph, source_name: str, load_name: str):
    """
    检查静态安全性（直通/Shoot-through），此时所有开关均为 OFF 状态。
    确保没有从电源（Source）到地（Ground）且绕过负载（Load）的非受控路径。
    """
    flow_graph = nx.DiGraph()
    flow_graph.add_nodes_from(graph.nodes())
    
    target_source = None
    
    # 构建流图 (Build flow graph)
    for u, v, key, data in graph.edges(keys=True, data=True):
        comp = data.get('component')
        if not comp: continue
        
        if comp.name == source_name:
            target_source = {'nodes': (u, v), 'obj': comp}
            continue
            
        if comp.name == load_name:
            # 标记负载边 (Mark load edges)
            flow_graph.add_edge(u, v, component=comp, weight=1.0, is_load=True)
            flow_graph.add_edge(v, u, component=comp, weight=1.0, is_load=True)
            continue
        
        # 根据组件类型分配权重 (Assign weights based on component type)
        if isinstance(comp, VoltageSource):
            flow_graph.add_edge(u, v, component=comp, weight=1.0)
            flow_graph.add_edge(v, u, component=comp, weight=1.0)
            
        elif isinstance(comp, (Wire, Inductor, Resistor)):
            flow_graph.add_edge(u, v, component=comp, weight=1.0)
            flow_graph.add_edge(v, u, component=comp, weight=1.0)
            
        elif isinstance(comp, Diode):
            n1, n2 = comp.nodes
            # 确保只添加正向导通的边 (Anode n1 -> Cathode n2)
            if u == n1 and v == n2:
                flow_graph.add_edge(u, v, component=comp, weight=1.0)
            else:
                flow_graph.add_edge(v, u, component=comp, weight=1.0)
                
        elif isinstance(comp, Switch):
            # 开关处于 OFF 状态。仅添加反向体二极管 (Body Diode): n2 -> n1
            n1, n2 = comp.nodes
            if u == n1 and v == n2:
                flow_graph.add_edge(v, u, component=comp, weight=1.0)
            else:
                flow_graph.add_edge(u, v, component=comp, weight=1.0)
                
    if not target_source:
        return True, "未找到电源 (Source not found)"
        
    u, v = target_source['nodes']
    vs = target_source['obj']
    pos, neg = vs.nodes
    
    try:
        # 检查从正极到负极的路径 (Check for path from Positive to Negative terminal)
        path = nx.shortest_path(flow_graph, source=pos, target=neg, weight='weight')
        
        # 用户修改：只要能找到路径就认为不安全
        path_str = _format_path_with_components(flow_graph, path)
        return False, f"静态漏电 (Static Leakage): 存在路径 {path_str} 导致电源 {vs.name} 泄漏"
            
    except nx.NetworkXNoPath:
        pass
        
    return True, "安全 (Safe)"

def analyze_switching_modes(graph: nx.MultiGraph, load_name: str):
    """
    枚举所有开关状态组合，并分析每种状态下的电路行为。
    返回每个状态的分析结果字典。
    """
    # 1. 识别电路中的所有开关 (Identify all switches)
    switches = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        comp = data.get('component')
        if isinstance(comp, Switch):
            switches.append({'edge': (u, v, key), 'name': comp.name, 'obj': comp})
            
    # 按名称排序以确保状态组合的一致性 (Sort by name for consistency)
    switches.sort(key=lambda x: x['name'])
    switch_names = [s['name'] for s in switches]
    
    results = {}
    num_switches = len(switches)
    # 2. 生成所有可能的开关状态组合 (Generate all state combinations)
    state_combinations = list(itertools.product([False, True], repeat=num_switches))
    
    for states in state_combinations:
        # 为当前状态创建一个新的图副本 (Create a graph copy for current state)
        mode_graph = graph.copy()
        
        # 3. 应用开关状态 (Apply switch states)
        for i, state in enumerate(states):
            sw_info = switches[i]
            u, v, key = sw_info['edge']
            
            if mode_graph.has_edge(u, v, key):
                edge_data = mode_graph.get_edge_data(u, v, key)
                new_comp = deepcopy(edge_data['component']) # 创建独立副本
                new_comp.state = state                      # 修改副本的状态 (True=ON, False=OFF)
                edge_data['component'] = new_comp           # 将边属性指向新的组件副本
        
        # 4. 检查动态短路 (Check for dynamic shorts)
        is_shorted, short_reason = check_dynamic_short(mode_graph)
        
        if is_shorted:
            # 如果发生短路，标记该模态无效
            analysis = {
                'valid': False,
                'is_shorted': True,
                'reasons': [short_reason],
                'inductor_trends': {}
            }
        else:
            # 5. 分析电感电压趋势 (Analyze inductor voltage trends)
            analysis = analyze_mode_characteristics(mode_graph, load_name)
        
        # 6. 存储结果 (Store results)
        results[states] = {
            'switches': switch_names,
            'graph': mode_graph,
            'valid': analysis['valid'],
            'is_shorted': analysis.get('is_shorted', False),
            'reasons': analysis['reasons'],
            'inductor_trends': analysis['inductor_trends']
        }
        
    return results

def check_dynamic_short(graph: nx.MultiGraph):
    """
    检查在当前开关状态下，是否存在电压源短路的情况。
    逻辑更新：
    1. 每个电压源分别作为"被测对象"。
    2. 计算静态电压分布（Static Voltage Analysis）：
       - "被测"电压源不作为电压源参与计算（视为开路，检测其两端电位差是由其他部分引起的）。
       - "其他"电压源正常参与计算，确立节点电位。
       - 导通开关视为短路。
    3. 根据计算出的节点电压，判断二极管（包括OFF开关的体二极管）是否反偏截止。
       - 如果 V_anode < V_cathode，则认为截止，不开通。
       - 否则认为可能导通，加入检测图。
    4. "其他"电压源在检测图中视为双向短路（低阻抗）。
    5. 检查"被测"电压源正负极是否连通。
    """
    import math
    
    # 找出所有电压源
    all_sources = []
    for u, v, key, data in graph.edges(keys=True, data=True):
        comp = data.get('component')
        if isinstance(comp, VoltageSource):
            all_sources.append({'nodes': (u, v), 'obj': comp})
            
    if not all_sources:
        return False, ""
        
    # 对每个电压源进行轮询检查
    for target in all_sources:
        target_vs = target['obj']
        t_pos, t_neg = target_vs.nodes
        
        # --- 步骤 1: 计算静态电压 (Static Analysis) ---
        # 构建用于计算电压的图
        voltage_calc_graph = nx.MultiGraph()
        voltage_calc_graph.add_nodes_from(graph.nodes())
        
        for u, v, key, data in graph.edges(keys=True, data=True):
            comp = data.get('component')
            if not comp: continue
            
            # 被测源不参与电压建立（我们想看外部电路给它施加了什么电压，或者是否短路）
            if comp == target_vs:
                continue
                
            voltage_calc_graph.add_edge(u, v, key=key, **data)
            
        # 计算节点电压
        # calculate_static_voltages 需要稍作修改以支持传入已知源，或者我们在外部预设好
        # 这里直接调用现有的 calculate_static_voltages，它会自动识别图中的 VoltageSource
        # 由于我们移除了 target_vs，它只会利用"其他"电源计算电压。
        # 注意：如果电路中没有"其他"电源（例如只有一个源），则所有节点电压为0（除非被地强驱），
        # 这时二极管偏置为 0V，通常视为可能导通（为了保守起见防止短路）。
        node_voltages = calculate_static_voltages(voltage_calc_graph)
        
        # --- 步骤 2: 构建短路检测流图 (Build Flow Graph) ---
        flow_graph = nx.DiGraph()
        flow_graph.add_nodes_from(graph.nodes())
        
        for u, v, key, data in graph.edges(keys=True, data=True):
            comp = data.get('component')
            if not comp: continue
            
            # A. 阻性/感性元件 (Resistor/Inductor/Wire) -> 视为低阻/通路
            # 注意：电感在瞬态短路分析中通常不视为短路路径（它限制di/dt），
            # 但如果只要有通路就算短路（稳态短路），则应加入。
            # 根据之前逻辑，Inductor 不加入（pass），Resistor/Wire 加入。
            if isinstance(comp, (Wire, Resistor)):
                flow_graph.add_edge(u, v, weight=0.1, component=comp)
                flow_graph.add_edge(v, u, weight=0.1, component=comp)
                
            # B. "其他" 电压源 -> 视为双向短路
            elif isinstance(comp, VoltageSource):
                if comp != target_vs:
                    flow_graph.add_edge(u, v, weight=0.1, component=comp)
                    flow_graph.add_edge(v, u, weight=0.1, component=comp)
                    
            # C. 开关 (Switch)
            elif isinstance(comp, Switch):
                if comp.state:
                    # ON -> 双向导通
                    flow_graph.add_edge(u, v, weight=0.1, component=comp)
                    flow_graph.add_edge(v, u, weight=0.1, component=comp)
                else:
                    # OFF -> 检查体二极管偏置
                    # 体二极管方向: Source(n2) -> Drain(n1)
                    n1, n2 = comp.nodes # n1=Drain, n2=Source
                    v_anode = node_voltages.get(n2, (0.0, False))[0]   # Source
                    v_cathode = node_voltages.get(n1, (0.0, False))[0] # Drain
                    
                    # 偏置检查: 仅当 V_anode >= V_cathode (或接近) 时加入
                    # 考虑到浮地等情况，如果是 Soft 驱动且电压相等，也加入（保守）
                    if v_anode >= v_cathode - 1e-6:
                        flow_graph.add_edge(n2, n1, weight=0.1, component=comp, type="body_diode")
                        
            # D. 二极管 (Diode)
            elif isinstance(comp, Diode):
                n1, n2 = comp.nodes # n1=Anode, n2=Cathode
                v_anode = node_voltages.get(n1, (0.0, False))[0]
                v_cathode = node_voltages.get(n2, (0.0, False))[0]
                
                if v_anode >= v_cathode - 1e-6:
                     flow_graph.add_edge(n1, n2, weight=0.1, component=comp)

        # --- 步骤 3: 检查路径 ---
        try:
            path = nx.shortest_path(flow_graph, source=t_pos, target=t_neg, weight='weight')
            path_str = _format_path_with_components(flow_graph, path)
            return True, f"短路 (Short Circuit): 电压源 {target_vs.name} 被短路，路径: {path_str}"
        except nx.NetworkXNoPath:
            pass
            
    return False, ""

def calculate_static_voltages(graph: nx.MultiGraph):
    """
    计算电路中的静态节点电压。
    假设节点0为参考地 (0V)。
    仅考虑电压源和导通的开关/导线。
    返回: {node_id: (voltage_float, is_hard_driven)}
    is_hard_driven: True 表示该节点被电压源通过低阻抗路径（导线/开关）强驱动；
                    False 表示该节点仅通过高阻抗/感性元件（电感/电阻）连接，电位较“软”，易被电感反电动势改变。
    """
    # 0: (Voltage, IsHard)
    voltages = {0: (0.0, True)} # 假设节点0是地，且是强驱动
    
    changed = True
    while changed:
        changed = False
        for u, v, data in graph.edges(data=True):
            comp = data.get('component')
            if not comp: continue
            
            # 1. 导通的开关/导线 (Low Impedance) -> 传播 Hard 属性
            if isinstance(comp, Wire) or (isinstance(comp, Switch) and comp.state):
                # 传播逻辑：如果源节点是 Hard，则目标节点也是 Hard。如果源是 Soft，目标也是 Soft。
                # 但如果目标已经是 Hard，则保持 Hard。
                
                # u -> v
                if u in voltages:
                    u_val, u_hard = voltages[u]
                    if v not in voltages:
                        voltages[v] = (u_val, u_hard)
                        changed = True
                    else:
                        v_val, v_hard = voltages[v]
                        # 如果 v 是 Soft 而 u 是 Hard，则升级 v 为 Hard
                        if not v_hard and u_hard:
                            voltages[v] = (u_val, True)
                            changed = True
                            
                # v -> u
                if v in voltages:
                    v_val, v_hard = voltages[v]
                    if u not in voltages:
                        voltages[u] = (v_val, v_hard)
                        changed = True
                    else:
                        u_val, u_hard = voltages[u]
                        if not u_hard and v_hard:
                            voltages[u] = (v_val, True)
                            changed = True

            # 2. 电感/电阻 (High Impedance / Inductive) -> 传播 Soft 属性
            elif isinstance(comp, (Inductor, Resistor)):
                # 无论源是 Hard 还是 Soft，通过电感/电阻传播后，目标都变成 Soft (除非目标已经被 Hard 驱动)
                
                # u -> v
                if u in voltages:
                    u_val, u_hard = voltages[u]
                    if v not in voltages:
                        voltages[v] = (u_val, False) # 降级为 Soft
                        changed = True
                    # 注意：如果 v 已经在 voltages 中，无论是 Hard 还是 Soft，我们都不覆盖。
                    # 因为 Hard > Soft，而 Soft vs Soft 也没必要更新（假设等电位）。
                    
                # v -> u
                if v in voltages:
                    v_val, v_hard = voltages[v]
                    if u not in voltages:
                        voltages[u] = (v_val, False) # 降级为 Soft
                        changed = True
                    
            # 3. 电压源 -> 保持 Hard/Soft 属性 (通常电压源本身是 Hard，但如果它串联在 Soft 路径中...)
            # 假设电压源本身是理想的，所以它保持驱动能力的“硬度”。
            elif isinstance(comp, VoltageSource):
                pos, neg = comp.nodes
                
                # neg -> pos
                if neg in voltages:
                    neg_val, neg_hard = voltages[neg]
                    target_val = neg_val + comp.dc_value
                    if pos not in voltages:
                        voltages[pos] = (target_val, neg_hard)
                        changed = True
                    else:
                        pos_val, pos_hard = voltages[pos]
                        if not pos_hard and neg_hard:
                            voltages[pos] = (target_val, True)
                            changed = True
                            
                # pos -> neg
                if pos in voltages:
                    pos_val, pos_hard = voltages[pos]
                    target_val = pos_val - comp.dc_value
                    if neg not in voltages:
                        voltages[neg] = (target_val, pos_hard)
                        changed = True
                    else:
                        neg_val, neg_hard = voltages[neg]
                        if not neg_hard and pos_hard:
                            voltages[neg] = (target_val, True)
                            changed = True
                    
    return voltages

def analyze_mode_characteristics(graph: nx.MultiGraph, load_name: str):
    """
    分析电感两端的电压，以确定电流变化趋势 (di/dt)。
    """
    valid = True
    reasons = []
    inductor_trends = {}
    
    # 0. 计算静态电压以判断二极管偏置 (Calculate static voltages)
    node_voltages = calculate_static_voltages(graph)
    # print(f"    [调试] 节点电压: {node_voltages}")
    
    flow_graph = nx.DiGraph()
    flow_graph.add_nodes_from(graph.nodes())
    
    # 1. 构建用于 KVL 分析的图 (Build graph for KVL analysis)
    for u, v, key, data in graph.edges(keys=True, data=True):
        comp = data.get('component')
        if not comp: continue
        
        if isinstance(comp, (VoltageSource, Wire, Capacitor, Resistor)):
            w = 0.1 if isinstance(comp, (Wire, Switch)) else 1.0
            flow_graph.add_edge(u, v, component=comp, weight=w)
            flow_graph.add_edge(v, u, component=comp, weight=w)
            
        elif isinstance(comp, Switch):
            if comp.state:
                # ON -> 导通
                flow_graph.add_edge(u, v, component=comp, weight=0.1)
                flow_graph.add_edge(v, u, component=comp, weight=0.1)
            else:
                # 开关 OFF -> 考虑反并联二极管 (Body Diode)
                # n2 -> n1 导通
                n1, n2 = comp.nodes
                
                # 检查偏置 (Check Bias)
                # Anode = n2, Cathode = n1
                
                v_anode_info = node_voltages.get(n2)
                v_cathode_info = node_voltages.get(n1)
                
                is_reverse_biased = False
                
                # 仅当两端电压都已知时才检查
                if v_anode_info and v_cathode_info:
                    v_anode, anode_hard = v_anode_info
                    v_cathode, cathode_hard = v_cathode_info
                    
                    # 核心逻辑：只有当两端都被“强驱动 (Hard)”时，才严格执行反偏截止。
                    # 如果任一端是“软驱动 (Soft)”（例如连接电感），则允许电感反电动势强行导通二极管。
                    if anode_hard and cathode_hard:
                        if v_anode < v_cathode - 0.1: # 留一点余量
                            is_reverse_biased = True
                        
                if not is_reverse_biased:
                    if u == n1 and v == n2: flow_graph.add_edge(v, u, component=comp, weight=0.1)
                    else: flow_graph.add_edge(u, v, component=comp, weight=0.1)
                    
        elif isinstance(comp, Diode):
            # 二极管：单向导通
            n1, n2 = comp.nodes # Anode=n1, Cathode=n2
            
            v_anode_info = node_voltages.get(n1)
            v_cathode_info = node_voltages.get(n2)
            
            is_reverse_biased = False
            
            if v_anode_info and v_cathode_info:
                v_anode, anode_hard = v_anode_info
                v_cathode, cathode_hard = v_cathode_info
                
                # 同上：仅在双端 Hard 时严格检查
                if anode_hard and cathode_hard:
                    if v_anode < v_cathode - 0.1:
                        is_reverse_biased = True
            
            if not is_reverse_biased:
                if u == n1 and v == n2: flow_graph.add_edge(u, v, component=comp, weight=0.1)
                else: flow_graph.add_edge(v, u, component=comp, weight=0.1)

    inductors = []
    for u, v, data in graph.edges(data=True):
        if isinstance(data['component'], Inductor):
            inductors.append({'nodes': (u, v), 'obj': data['component']})
            
    # 2. 对每个电感进行分析 (Analyze each inductor)
    for item in inductors:
        ind = item['obj']
        n1, n2 = ind.nodes
        
        # 寻找闭合回路
        try:
            path = nx.shortest_path(flow_graph, source=n2, target=n1, weight='weight')
        except nx.NetworkXNoPath:
            try:
                path_rev = nx.shortest_path(flow_graph, source=n1, target=n2, weight='weight')
                path = path_rev
            except nx.NetworkXNoPath:
                path = None
                
        if path is None:
            valid = False
            reasons.append(f"电感 {ind.name} 开路 (Open Circuit): 无法找到闭合回路")
            inductor_trends[ind.name] = "开路 (Open Circuit)"
            continue
            
        # 3. 验证负载充电方向 (Verify Load Charging Direction)
        path_valid = True
        for i in range(len(path) - 1):
            u_node, v_node = path[i], path[i+1]
            edge_data = flow_graph.get_edge_data(u_node, v_node)
            comp = edge_data['component']
            
            # 识别负载组件
            is_load = False
            if comp.name == load_name:
                is_load = True
            elif isinstance(comp, Resistor): 
                is_load = True
            elif isinstance(comp, VoltageSource) and getattr(comp, 'role', '') == 'output': 
                is_load = True
            
            if is_load:
                if isinstance(comp, VoltageSource):
                    pos, neg = comp.nodes
                    if u_node == neg and v_node == pos:
                        path_str = _format_path_with_components(flow_graph, path)
                        # print(f"    [无效路径] 电感 {ind.name} 导致负载 {comp.name} 放电 (路径: {path_str})")
                        path_valid = False
                        break
        
        if not path_valid:
            valid = False
            reasons.append(f"电感 {ind.name} 路径导致负载放电 (Discharges Load)")
            inductor_trends[ind.name] = "无效 (负载放电)"
            continue

        # 4. 计算路径上的电压降
        v_drop_total = 0.0
        for i in range(len(path) - 1):
            u_node, v_node = path[i], path[i+1]
            edge_data = flow_graph.get_edge_data(u_node, v_node)
            comp = edge_data['component']
            
            if isinstance(comp, VoltageSource):
                pos, neg = comp.nodes
                if u_node == pos and v_node == neg: v_drop_total += comp.dc_value
                elif u_node == neg and v_node == pos: v_drop_total -= comp.dc_value
            
        # 5. 计算电感电压
        if path[0] == n1: v_ind = v_drop_total
        else: v_ind = -v_drop_total
        
        # path_str = _format_path_with_components(flow_graph, path)
        # print(f"    [电感 {ind.name}] 回路: {path_str}, 电压: {v_ind:.2f}V")
        
        # 6. 判断电流趋势
        if v_ind > 1e-3: trend = "电流增加 (Increasing)"
        elif v_ind < -1e-3: trend = "电流减小 (Decreasing)"
        else: trend = "电流恒定 (Constant)"
            
        inductor_trends[ind.name] = trend
        
    return {
        'valid': valid,
        'reasons': reasons,
        'inductor_trends': inductor_trends
    }
