import networkx as nx
import sys
from collections import deque
import re
from math import inf

sys.setrecursionlimit(1000000)

#BUILD BIEDGED GRAPH AS AN UNDIRECTED GRAPH HAVING ALL EDGES ++ +- -- -+ INTO ACCOUNT
def parse_and_build_bi_graph(file_path):
    """
    Parses a GFA and builds a bidirected graph:
      • segment-edges (black) between node_L and node_R (with increased width)
      • link-edges (gray) between the correct ends depending on +/-.
    """
    G = nx.Graph()

    # 1) first pass: collect all segment IDs
    segments = []
    with open(file_path) as f:
        for line in f:
            if not line or line[0] != 'S':
                continue
            parts = line.split()
            segments.append(parts[1])

    # 2) add each segment’s two ends as nodes and a wide black edge between them
    for seg in segments:
        G.add_node(f"{seg}_L")
        G.add_node(f"{seg}_R")
        G.add_edge(f"{seg}_L", f"{seg}_R", color="black", width=10)

    # 3) second pass: add each link with the correct ends
    with open(file_path) as f:
        for line in f:
            if not line or line[0] != 'L':
                continue
            parts = line.split()
            _, from_id, from_orient, to_id, to_orient, *_ = parts

            # choose end-labels by orientation
            u = f"{from_id}_{'R' if from_orient == '+' else 'L'}"
            v = f"{to_id}_{'L' if to_orient   == '+' else 'R'}"

            # sanity check—only add if both ends actually exist
            if not G.has_node(u) or not G.has_node(v):
                print(f"Warning: skipping invalid link {from_id}{from_orient}->{to_id}{to_orient}")
                continue

            G.add_edge(u, v, color="gray")

    # 4) summary of graph contents
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    print(f"Added {num_nodes} nodes and {num_edges} edges.")

    return G

#THEY DEFINE THE TRAVERSAL RULES USED TO FOLLOW THE WORKING ROOT AND? CALCULATE THE DISTANCES FROM THE ROOT
def allowed(u, v, data):
    """
    Return True iff an edge u→v with attributes `data` is traversable.
      • Black edges only go L→R
      • Gray edges only go R→L
      • All other edges are disallowed
    """
    color = data.get("color")
    return allowed_edge(u, v, color)

#IT IS USED TO THE FIND ROOT TO CALCULATE THE DISTANCES 
def get_allowed_bfs_distances(graph, root):
    """
    Computes allowed distances from the given root on the full graph,
    using our allowed() rules via node suffixes and edge 'color'.
    Neighbors are processed in sorted order for stability.

    Returns a dict mapping each reachable node to its BFS distance.
    """
    distances = {root: 0}
    queue = deque([root])
    while queue:
        current = queue.popleft()
        for neighbor in sorted(graph[current].keys()):
            edge_data = graph[current][neighbor]
            if (neighbor not in distances
                and allowed(current, neighbor, edge_data)
            ):
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances

#IT IS USEDD TO FIND DISTANCES FROM THE ROOT TO SORT THE SNARLS AS R-L ETC
def check_traversal_order(graph, node1, node2, root):
    """
    Given two node IDs, order them so the one closer (by allowed BFS) to root comes first.
    If equal distance or unreachable, preserves original order.
    """
    distances = get_allowed_bfs_distances(graph, root)
    d1 = distances.get(node1, inf)
    d2 = distances.get(node2, inf)
    return (node1, node2) if d1 <= d2 else (node2, node1)

#IT DEFINES THE EDGE TRAVERSAL RULES IN PRECOMP FORWARD 
def allowed_edge(u, v, color):
    """
    Returns True if the edge (u, v) is allowed under VG rules:
      - black edges only go L→R
      - gray  edges only go R→L
    """
    if color == "black":
        return u.endswith("_L") and (v == u.replace("_L", "_R"))
    elif color == "gray":
        return u.endswith("_R") and v.endswith("_L")
    return False

#FIND SOURCE AND SINK SESSION
#FIND ROOT AND CHECK BY FOLLOWING THE TRAVERSAL RULES IF ALL NODES ARE ACCESIBLE
def find_root(graph):
    # 0) ensure any self‐edges are colored black
    for u, v, data in graph.edges(data=True):
        if u.split('_',1)[0] == v.split('_',1)[0]:
            data['color'] = 'black'

    # 1) detect & print any "bidirectional" gray edges (LL or RR) —
    bidir = []
    for u, v, data in graph.edges(data=True):
        if data.get("color") != "gray":
            continue
        # skip actual segment‐edges
        if u.split('_',1)[0] == v.split('_',1)[0]:
            continue
        # same suffix = LL or RR
        if u.endswith("_L") and v.endswith("_L") or u.endswith("_R") and v.endswith("_R"):
            bidir.append((u, v))
    #print("Bidirectional gray‐edge pairs (forbidden LL/RR):", bidir)
    print("Bidirectional gray-edge pairs (forbidden LL/RR NUMBER):", len(bidir))
    # 2) build in‐degree only over non‐segment edges
    in_deg = {n: 0 for n in graph.nodes()}
    for u, v, data in graph.edges(data=True):
        # skip segment‐edges entirely for root‐finding
        if u.split('_',1)[0] == v.split('_',1)[0]:
            continue

        if allowed(u, v, data):
            in_deg[v] += 1
        if allowed(v, u, data):
            in_deg[u] += 1

    # 3) pick zero‐in‐degree leaf‐Ls
    candidates = sorted(n for n, d in in_deg.items() if d == 0 and n.endswith("_L"))
    print("\nZero-in-degree leaf-L candidates:", candidates)

    total = graph.number_of_nodes()
    # 4) test reachability under your allowed() BFS
    for r in candidates:
        reached = len(get_allowed_bfs_distances(graph, r))
        if reached == total:
            print(f"Chosen root: {r} - reaches all {total} nodes.")
            return r
        else:
            print(f"Candidate {r} reaches {reached}/{total} nodes.")
    print("No working root found.")
    return None

#FINDING THE SINK BUT LEXIGOCRAPHICALLY
def find_end(graph):
    """Finds the end node in a bi-edged graph, ensuring correct lexicographic ordering."""
    def sort_key(node):
        match = re.match(r"([A-Za-z]*)(\d*)(_L|_R)?$", node)
        if not match:
            # Place unexpected formats at the end
            return ("", float('-inf'), "~")
        prefix, num, suffix = match.groups()
        num = int(num) if num.isdigit() else float('inf')
        return (prefix, num, suffix or "")
 
    return max(graph.nodes(), key=sort_key)
