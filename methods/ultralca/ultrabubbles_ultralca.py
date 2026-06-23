import sys

sys.setrecursionlimit(1000000)

from pipeline import lca
from pipeline.common import pair_custom_key

# UltraLCA
def find_ultrabubbles_ultralca(graph, snarls, ftip):
    ultrabubbles = set()

    for x, y in snarls:
        # Only allow pairs with r,l orientation.
        if not (x.endswith("_R") and y.endswith("_L")):
            continue
        outer_break = True

        for node in ftip:
            if lca.find_lca_forward(node, x) == x and lca.find_lca_forward(node, y) != y:
                outer_break = False
                break
        if outer_break is False:
            continue

        if outer_break is True:
            ultrabubbles.add((x, y))
    return sorted(ultrabubbles, key=pair_custom_key)
