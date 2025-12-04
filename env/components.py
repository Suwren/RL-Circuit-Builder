from dataclasses import dataclass, field
from typing import Tuple, Optional

@dataclass
class Component:
    """
    电路元件基类。
    
    属性:
        name: 元件名称 (如 'R1', 'C1').
        nodes: 连接的节点索引元组 (n1, n2).
        value: 元件值 (电阻/电容/电感值).
    """
    name: str
    nodes: Tuple[int, int]
    value: float = 0.0

@dataclass
class Inductor(Component):
    """
    电感元件。
    
    属性:
        ic: 初始电流 (Initial Current)，用于仿真。
    """
    ic: float = 0.0

@dataclass
class Capacitor(Component):
    """
    电容元件。
    
    属性:
        ic: 初始电压 (Initial Voltage)，用于仿真。
    """
    ic: float = 0.0

@dataclass
class Resistor(Component):
    """电阻元件。"""
    pass

@dataclass
class VoltageSource(Component):
    """
    电压源元件。
    
    属性:
        dc_value: 直流电压值。
        ac_amplitude: 交流幅值 (可选)。
        frequency: 频率 (可选)。
    """
    dc_value: float = 0.0
    ac_amplitude: float = 0.0
    frequency: float = 0.0

@dataclass
class CurrentSource(Component):
    """
    电流源元件。
    
    属性:
        dc_value: 直流电流值。
    """
    dc_value: float = 0.0

@dataclass
class Switch(Component):
    """
    理想开关元件 (MOSFET 模型)。
    
    属性:
        state: 开关状态 (True=闭合/导通, False=断开/截止)。
    """
    state: bool = False # False = Open (OFF), True = Closed (ON)

@dataclass
class Diode(Component):
    """
    二极管元件。
    """
    pass

@dataclass
class Wire(Component):
    """
    理想导线 (零电阻)。
    """
    pass
    name: str
    nodes: Tuple[int, int]  # (节点A, 节点B)
    value: float = 0.0
    
    @property
    def type_name(self) -> str:
        return self.__class__.__name__

@dataclass
class Inductor(Component):
    """
    电感元件 (Inductor)
    value: 电感值，单位亨利 (H)
    ic: 初始电流 (Initial Current)，用于检测断路
    """
    ic: float = 0.0

@dataclass
class Capacitor(Component): 
    """
    电容元件 (Capacitor)
    value: 电容值，单位法拉 (F)
    """
    pass

@dataclass
class Resistor(Component): 
    """
    电阻元件 (Resistor)
    value: 电阻值，单位欧姆 (Ohm)
    """
    pass

@dataclass
class VoltageSource(Component):
    """
    电压源 (Voltage Source)
    dc_value: 直流电压值，单位伏特 (V)
    ac_amplitude: 交流幅值 (暂未使用)
    frequency: 频率 (暂未使用)
    """
    dc_value: float = 0.0
    ac_amplitude: float = 0.0
    frequency: float = 0.0

@dataclass
class Switch(Component):
    """
    开关 (Switch)
    state: 初始状态 (False=断开, True=闭合)
    """
    state: bool = False 

@dataclass
class Diode(Component):
    """
    二极管 (Diode)
    model: SPICE模型名称 (默认为 "D")
    """
    model: str = "D"

@dataclass
class CurrentSource(Component):
    """
    电流源 (Current Source)
    dc_value: 直流电流值 (A)
    """
    dc_value: float = 0.0

@dataclass
class Wire(Component):
    """
    导线 (Wire)
    """
    pass
