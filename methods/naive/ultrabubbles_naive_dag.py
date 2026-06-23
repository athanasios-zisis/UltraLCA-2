import sys
from collections import deque

sys.setrecursionlimit(1000000)

from pipeline.common import pair_custom_key

#FIND ULTRABUBBLES WITH THÎ• NAIVE DEFÎ™ÎÎ™Î¤Î™ÎŸÎ APPROACH BUT CHECKING ONLY TIPS SINCE THE INPUT ARE DAG

#COMPUTE ULTRABUBBLES BY THE NAIVE APPROACH


def find_ultra(graph, snarls, quiet: bool = False):
    ultrabubbles = []
    counter=0
    for node1, node2 in snarls:
        if not (node1.endswith("_R") and node2.endswith("_L")):
            counter+=1
            continue 
        # 1) BFS with pruning of black edges incident on node1/node2
        comp = set()
        q = deque([node1])
        while q:
            u = q.popleft()
            if u in comp:
                continue
            comp.add(u)
            for v, data in graph[u].items():
                # skip only black edges touching the endpoints
                if data.get("color") == "black" and {u, v} & {node1, node2}:
                    continue
                if v not in comp:
                    q.append(v)
    
        # must reach node2
        if node2 not in comp:
            continue
    
        # 2) leaf check in the pruned component
        leaf = False
        for n in comp - {node1, node2}:
            deg_in_pruned = 0
            for v, data in graph[n].items():
                if v not in comp:
                    continue
                if data.get("color") == "black" and {n, v} & {node1, node2}:
                    continue
                deg_in_pruned += 1
                if deg_in_pruned > 1:
                    break
            if deg_in_pruned == 1:
                leaf = True
                break
        if leaf:
            continue
    
        # 3) mark ultra
        ultrabubbles.append((node1, node2))
    
    ultrabubbles.sort(key=pair_custom_key)
    if not quiet:
        print(f'COUNTER NAIVE L-R:', counter)
    return ultrabubbles
