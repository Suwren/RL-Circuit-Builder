from dataclasses import dataclass, field
from typing import Tuple, Optional

@dataclass
class Component:
    """
    Base class for all circuit components.
    """
    name: str
    nodes: Tuple[int, int]
    value: float = 0.0

    def get_type_id(self) -> int:
        """
        Returns a unique integer ID for the component type.
        Used for constructing the observation tensor.
        """
        c_name = self.__class__.__name__
        # Simplified Type IDs for Reduced Adjacency Matrix
        # 1: VoltageSource
        # 2: Inductor
        # 3: Switch
        # 0: Others (Wire, Capacitor, Resistor, Diode, CurrentSource) - Treated as empty/irrelevant for this specific input model
        if c_name == "VoltageSource": return 1
        if c_name == "Inductor": return 2
        if c_name == "Switch": return 3
        return 0

@dataclass
class Inductor(Component):
    """Inductor component."""
    ic: float = 0.0 # Initial Current

@dataclass
class Capacitor(Component):
    """Capacitor component."""
    ic: float = 0.0 # Initial Voltage

@dataclass
class Resistor(Component):
    """Resistor component."""
    pass

@dataclass
class VoltageSource(Component):
    """Voltage Source (DC/AC)."""
    dc_value: float = 0.0
    ac_amplitude: float = 0.0
    frequency: float = 0.0
    role: str = "general" # Options: "input", "output", "general"

@dataclass
class CurrentSource(Component):
    """Current Source."""
    dc_value: float = 0.0

@dataclass
class Switch(Component):
    """Ideal Switch (MOSFET model)."""
    state: bool = False # False=OFF, True=ON

@dataclass
class Diode(Component):
    """Ideal Diode."""
    pass

@dataclass
class Wire(Component):
    """Ideal Wire (Zero Resistance)."""
    pass
    name: str
    nodes: Tuple[int, int]
    value: float = 0.0
    
    @property
    def type_name(self) -> str:
        return self.__class__.__name__
