import networkx as nx
import os
import sys
from pathlib import Path
import PySpice.Logging.Logging as Logging
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *
from env.components import Component, Inductor, Capacitor, Resistor, VoltageSource, Switch, Diode, Wire, CurrentSource

# 尝试为Windows上的Conda环境自动设置Ngspice库路径
if os.name == 'nt':
    # 检查是否在conda环境中
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        ngspice_path = Path(conda_prefix) / "Library" / "bin" / "ngspice.dll"
        if ngspice_path.exists():
            # print(f"Found Ngspice at: {ngspice_path}")
            os.environ["PYSPICE_NGSPICE_LIBRARY_PATH"] = str(ngspice_path)

# 配置 PySpice 日志级别
logger = Logging.setup_logging()

class CircuitSimulator:
    """
    基于 PySpice (Ngspice) 的电路仿真器。
    负责将 NetworkX 图转换为 SPICE 网表并运行仿真。
    """
    
    def __init__(self):
        pass
        
    def run_snapshot(self, graph: nx.MultiGraph, uic=True):
        """
        运行快照仿真 (极短时间的瞬态仿真)。
        用于获取电路在特定开关状态下的瞬时电压和电流。
        
        参数:
            graph: 电路图。
            uic: 是否使用初始条件 (Use Initial Conditions)。
                 True = 瞬态分析从 t=0 开始，考虑电感/电容初始值。
                 False = 先计算 DC 工作点 (通常用于纯电阻/DC分析)。
        """
        # 运行 1ns 的瞬态仿真
        return self.run_transient(graph, step_time=1e-10, end_time=1e-9, use_initial_condition=uic)

    def run_transient(self, graph: nx.MultiGraph, step_time, end_time, use_initial_condition=True):
        """
        运行瞬态仿真。
        """
        circuit = self._build_circuit(graph)
        
        # 创建仿真器对象
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        
        try:
            # 运行仿真
            # use_initial_condition=True 对应 SPICE 的 'uic' 参数
            # 这对于包含电感/电容且需要指定初始状态的电路至关重要
            analysis = simulator.transient(step_time=step_time, end_time=end_time, use_initial_condition=use_initial_condition)
            return analysis
        except Exception as e:
            # 仿真失败 (如矩阵奇异、不收敛等)
            # print(f"Simulation failed: {e}") 
            return None

    def _build_circuit(self, graph: nx.MultiGraph) -> Circuit:
        """
        将 NetworkX 图转换为 PySpice Circuit 对象。
        """
        circuit = Circuit('RL_Generated_Circuit')
        
        # 添加 SPICE 模型 (如二极管模型)
        circuit.model('D', 'D', is_=1e-14, rs=0.1) # 通用二极管模型
        
        nodes = set()
        
        # 遍历图中的边并添加元件
        for u, v, data in graph.edges(data=True):
            comp = data['component']
            name = comp.name
            n1 = str(u)
            n2 = str(v)
            nodes.add(n1)
            nodes.add(n2)
            
            if isinstance(comp, Resistor):
                circuit.Resistor(name, n1, n2, comp.value@u_Ohm)
                
            elif isinstance(comp, Inductor):
                # 添加电感，并设置初始电流 (ic)
                circuit.Inductor(name, n1, n2, comp.value@u_H, ic=comp.ic@u_A)
                
            elif isinstance(comp, Capacitor):
                # 添加电容，并设置初始电压 (ic)
                circuit.Capacitor(name, n1, n2, comp.value@u_F, ic=comp.ic@u_V)
                
            elif isinstance(comp, VoltageSource):
                circuit.VoltageSource(name, n1, n2, comp.dc_value@u_V)
                
            elif isinstance(comp, CurrentSource):
                # 注意: CurrentSource 方向是从 n1 流向 n2 (在源内部)
                circuit.CurrentSource(name, n1, n2, comp.dc_value@u_A)
                
            elif isinstance(comp, Switch):
                # 开关模型:
                # 闭合 (ON): 小电阻 (0.001 Ohm)
                # 断开 (OFF): 大电阻 (1G Ohm) + 并联体二极管
                
                if comp.state: # ON
                    r_val = 0.001
                    circuit.Resistor(name, n1, n2, r_val@u_Ohm)
                else: # OFF
                    r_val = 1e9
                    circuit.Resistor(name, n1, n2, r_val@u_Ohm)
                    
                # 始终添加体二极管 (Body Diode)
                # 假设是 NMOS: Source(n2) -> Drain(n1)
                # 阳极接 n2, 阴极接 n1
                circuit.Diode(f"body_{name}", n2, n1, model='D')
                
            elif isinstance(comp, Diode):
                circuit.Diode(name, n1, n2, model='D')
                
            elif isinstance(comp, Wire):
                # 导线建模为极小电阻
                circuit.Resistor(name, n1, n2, 1e-6@u_Ohm)
            
        # 添加接地电阻 (Grounding Resistors)
        # 为了防止 SPICE 出现 "Floating Node" (悬空节点) 错误，
        # 我们在每个节点和地之间连接一个极大的电阻 (1G Ohm)。
        # 这对电路行为影响极小，但能显著提高仿真稳定性。
        for node in nodes:
            if node != '0':
                circuit.Resistor(f"R_gnd_{node}", node, '0', 1e9@u_Ohm)
                
        return circuit
