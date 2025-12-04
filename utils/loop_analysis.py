import networkx as nx
import numpy as np
from env.components import VoltageSource

def find_loops_and_direction(graph: nx.MultiGraph):
    """
    Finds all simple cycles in the circuit graph and determines the current direction
    based on the net voltage from VoltageSources.
    
    Args:
        graph: The circuit graph (nx.MultiGraph)
        
    Returns:
        List of dictionaries, each containing:
        - 'path': List of nodes in the loop (ordered)
        - 'edges': List of edge data dictionaries in the loop
        - 'net_voltage': Net voltage in the direction of the path
        - 'direction': "Clockwise" (relative to path) or "Counter-Clockwise" (if net_voltage < 0)
                       Actually, we just return the corrected path order.
    """
    # 1. Find Cycle Basis (Fundamental Cycles)
    # Note: cycle_basis finds a basis, not ALL simple cycles. 
    # For a planar circuit like Buck, basis is usually the meshes.
    # simple_cycles is for directed graphs.
    # For undirected, we can use simple_cycles on a directed version or use cycle_basis.
    # Let's use simple_cycles on a conversion to DiGraph to find ALL simple cycles if the graph is small.
    # Since our graph is small (<20 nodes), this is feasible.
    
    # Convert to directed to find all cycles (treating edges as bidirectional initially)
    # But wait, simple_cycles on a bidirectional graph will return 2-cycles (A->B->A). We want geometric loops.
    # Better approach: Use nx.cycle_basis(G) to get fundamental loops, then maybe combine them?
    # Or just use cycle_basis as it usually gives the "meshes" for planar graphs.
    
    # However, MultiGraph support in cycle_basis is tricky. It ignores parallel edges.
    # We need to handle parallel edges manually if any.
    # For the Buck example (Vin, S, D, L, Vout), there are no parallel edges between same pair of nodes.
    # So cycle_basis is safe for now.
    
    G_simple = nx.Graph(graph) # Convert to simple graph for cycle finding
    cycles = nx.cycle_basis(G_simple)
    
    results = []
    
    for cycle_nodes in cycles:
        # cycle_nodes is a list of nodes, e.g., [0, 1, 2]
        # We need to reconstruct the edges and calculate voltage.
        
        # Form the closed loop path: append start node to end
        path = cycle_nodes + [cycle_nodes[0]]
        
        net_voltage = 0.0
        edges_data = []
        
        # Traverse the loop
        for i in range(len(cycle_nodes)):
            u = path[i]
            v = path[i+1]
            
            # Find the edge between u and v in the MultiGraph
            # Note: If there are parallel edges, this logic simply picks one. 
            # To be robust, we should iterate all parallel edges, but that explodes complexity (loops * parallel_combinations).
            # For this task, we assume one component per connection or pick the first one.
            edge_data = graph.get_edge_data(u, v)[0] # Pick key 0
            edges_data.append(edge_data)
            
            comp = edge_data['component']
            
            # Check Voltage Source
            if isinstance(comp, VoltageSource):
                # Voltage Source direction: nodes[0] (+) -> nodes[1] (-)
                # If we go u -> v:
                # If u == nodes[1] and v == nodes[0]: We go (-) to (+), GAIN Voltage (+V)
                # If u == nodes[0] and v == nodes[1]: We go (+) to (-), DROP Voltage (-V)
                
                if u == comp.nodes[1] and v == comp.nodes[0]:
                    net_voltage += comp.dc_value
                elif u == comp.nodes[0] and v == comp.nodes[1]:
                    net_voltage -= comp.dc_value
        
        # Determine Direction
        # If net_voltage > 0: The current flows in the direction of our traversal (path).
        # If net_voltage < 0: The current flows opposite to our traversal.
        
        final_path = path
        if net_voltage < 0:
            final_path = path[::-1] # Reverse the path
            net_voltage = -net_voltage
            
        results.append({
            "nodes": final_path,
            "net_voltage": net_voltage,
            "components": [e['component'].name for e in edges_data]
        })
        
    return results
