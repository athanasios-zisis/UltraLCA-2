import networkx as nx
import matplotlib.pyplot as plt
import sys
import re

sys.setrecursionlimit(1000000)

#THEY ARE USED FOR ORDERING OUTPUTS LIKE TIPS AND CYCLENODES AND SORTING PAIRS
def node_custom_key(node):
    """
    Sort key for a single node:
      - If the name begins with digits, key = (0, number, suffix)
        where suffix is 0 for "L" and 1 for "R".
      - Otherwise, key = (1, node_string, 0).
    This ensures numeric IDs sort in numeric order, with "L" before "R".
    """
    m = re.match(r"(\d+)_?([LR]?)$", node)
    if m:
        num = int(m.group(1))
        suffix = m.group(2)
        orient = 0 if suffix == "L" else 1
        return (0, num, orient)
    else:
        return (1, node.lower(), 0)

def pair_custom_key(pair):
    """
    Sort key for a pair of nodes, by applying node_custom_key to each.
    """
    return (node_custom_key(pair[0]), node_custom_key(pair[1]))

#EXTRA THAT COULD BE USED TO VISUALIZE A (SMALL) BIGRAPH 
#BUT THE FOLLOWING IPMORT IS NEEDED:import matplotlib.pyplot as plt
def visualize_bi_graph(graph):
    pos = nx.spring_layout(graph)
    colors = [data["color"] for _, _, data in graph.edges(data=True)]
    nx.draw(graph, pos, with_labels=True, edge_color=colors, node_size=500, font_size=10)
    plt.show()