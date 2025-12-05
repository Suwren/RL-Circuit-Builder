import matplotlib.pyplot as plt
import networkx as nx
from env.components import VoltageSource, Switch, Inductor, Capacitor, Diode

def plot_circuit(graph, filename="generated_topology.png"):
    """
    Visualizes the circuit topology using NetworkX and Matplotlib.
    Draws components with directional arrows where applicable.
    """
    plt.figure(figsize=(12, 10))
    pos = nx.circular_layout(graph)
    
    # Draw Nodes
    nx.draw_networkx_nodes(graph, pos, node_size=800, node_color='lightblue', edgecolors='black')
    nx.draw_networkx_labels(graph, pos, font_size=12, font_weight='bold')
    
    edge_labels = {}
    
    # Draw Edges (Components)
    for u, v, key, data in graph.edges(keys=True, data=True):
        comp = data.get('component')
        if not comp: continue
        
        label = comp.name
        color = 'black'
        style = 'solid'
        arrow_style = '-|>'
        
        # Customize appearance based on component type
        if isinstance(comp, VoltageSource):
            label += "\n(+ -> -)"
            color = 'red'
            arrow_style = '-|>'
        elif isinstance(comp, Switch):
            label += "\n(D -> S)"
            color = 'green'
        elif isinstance(comp, Diode):
            label += "\n(A -> K)"
            color = 'orange'
        elif isinstance(comp, Inductor):
            color = 'blue'
        elif isinstance(comp, Capacitor):
            color = 'purple'
            
        if (u, v) in edge_labels:
            edge_labels[(u, v)] += f"\n{label}"
        else:
            edge_labels[(u, v)] = label
            
        nx.draw_networkx_edges(
            graph, pos, 
            edgelist=[(u, v)], 
            width=2, 
            edge_color=color, 
            style=style,
            arrows=True,
            arrowstyle=arrow_style,
            arrowsize=20,
            connectionstyle=f"arc3,rad=0.1" # Curved edges for parallel components
        )
            
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=9, label_pos=0.5)
    
    plt.title("AlphaZero Generated Topology (with Directions)", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Topology plot saved to: {filename}")
    plt.close()
