import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def visualize_circuit(graph: nx.MultiGraph, title="Generated Circuit", filename=None):
    """
    使用 matplotlib 可视化电路图，包含元件方向性和详细图例。
    """
    plt.figure(figsize=(14, 10)) # 稍微加宽以容纳图例
    
    # 布局
    pos = nx.spring_layout(graph, seed=42, k=2.5)
    
    # 绘制节点
    node_colors = ['lightgray' if n == 0 else 'lightblue' for n in graph.nodes()]
    nx.draw_networkx_nodes(graph, pos, node_size=800, node_color=node_colors, edgecolors='black')
    nx.draw_networkx_labels(graph, pos, font_size=12, font_weight='bold')
    
    # 跟踪边数量以进行偏移
    edge_counts = {}
    
    ax = plt.gca()
    
    # 图例句柄
    legend_handles = []
    added_labels = set()
    
    for u, v, data in graph.edges(data=True):
        comp = data.get('component')
        if not comp: continue
        
        n1, n2 = comp.nodes
        
        # 确定颜色和标签
        color = 'black'
        label_text = comp.name
        type_name = type(comp).__name__
        
        if "VoltageSource" in str(type(comp)):
            color = 'red'
            label_text += "\n(+)->(-)" 
            legend_label = "Voltage Source (Arrow: + to -)"
        elif "Switch" in str(type(comp)):
            color = 'green'
            # 假设 n1=Drain, n2=Source
            # 箭头方向: n1 -> n2 (Drain -> Source)
            # 体二极管方向: n2 -> n1 (Source -> Drain, 反向)
            label_text += "\n(D->S)" 
            legend_label = "Switch (Arrow: Drain->Source)\n(Body Diode: Source->Drain)"
        elif "Diode" in str(type(comp)):
            color = 'blue'
            label_text += "\n(A->K)"
            legend_label = "Diode (Arrow: Anode->Cathode)"
        elif "Inductor" in str(type(comp)):
            color = 'orange'
            legend_label = "Inductor"
        elif "Capacitor" in str(type(comp)):
            color = 'purple'
            legend_label = "Capacitor"
        else:
            legend_label = type_name
            
        # 添加图例 (去重)
        if legend_label not in added_labels:
            legend_handles.append(mpatches.Patch(color=color, label=legend_label))
            added_labels.add(legend_label)
            
        # 计算平行边的偏移
        pair = tuple(sorted((n1, n2)))
        count = edge_counts.get(pair, 0)
        edge_counts[pair] = count + 1
        
        # 偏移逻辑
        # 垂直于线移动
        p1 = np.array(pos[n1])
        p2 = np.array(pos[n2])
        vec = p2 - p1
        length = np.linalg.norm(vec)
        if length == 0: continue # 自环?
        
        unit_vec = vec / length
        perp_vec = np.array([-unit_vec[1], unit_vec[0]])
        
        # 偏移量 (交替侧)
        shift_scale = 0.15 * ((count + 1) // 2) * (-1 if count % 2 else 1)
        
        start = p1 + perp_vec * shift_scale
        end = p2 + perp_vec * shift_scale
        
        # 绘制箭头
        # shrinkA/B 以避免覆盖节点
        ax.annotate("",
                    xy=end, xycoords='data',
                    xytext=start, textcoords='data',
                    arrowprops=dict(arrowstyle="->", color=color, lw=2, shrinkA=15, shrinkB=15, 
                                    connectionstyle="arc3,rad=0.05")) # 轻微弯曲看起来更好
                                    
        # 在中点绘制标签
        mid = (start + end) / 2
        ax.text(mid[0], mid[1], label_text, color=color, fontsize=9, ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    plt.title(title)
    plt.axis('off')
    
    # 添加图例
    if legend_handles:
        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1), title="Component Legend")
        plt.tight_layout() # 调整布局以显示图例
    
    if filename:
        plt.savefig(filename, bbox_inches='tight') # 确保图例被保存
        print(f"Circuit visualization saved to {filename}")
    else:
        plt.show()
