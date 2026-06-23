import sys
import math
import time
from collections import deque
from collections import defaultdict

sys.setrecursionlimit(1000000)

from pipeline.parse_root import allowed_edge


# THEY DEFINE TH TRAVERSAL RULES USED TO FOLLOW THE WORKING ROOT AND??? SORT JSON SNARLS FROM THE ROOT?
def allowed(u, v, data):
    """
    Return True iff an edge u→v with attributes `data` is traversable.
      • Black edges only go L→R
      • Gray edges only go R→L
      • All other edges are disallowed
    """
    color = data.get("color")
    return allowed_edge(u, v, color)


# LCA QUERIES BY RMQ EULER APROACH
# ─────────────────────────────────────────────────────────
# Linear-time RMQ structure for ±1-difference arrays
class RMQLinear:
    def __init__(self, D):
        """
        D: list of ints where |D[i+1] - D[i]| = 1
        """
        self.D = D
        n = len(D)
        # block size = max(1, floor(½ log2 n))
        b = max(1, int(math.log2(n) // 2))
        m = (n + b - 1) // b
        self.b = b
        self.n = n

        # 1) block minima positions
        block_mins = [0] * m
        for bi in range(m):
            l = bi * b
            r = min(l + b, n)
            mi = l
            for j in range(l + 1, r):
                if D[j] < D[mi]:
                    mi = j
            block_mins[bi] = mi

        # 2) sparse-table on block_mins
        log = [0] * (m + 1)
        for i in range(2, m + 1):
            log[i] = log[i // 2] + 1
        K = log[m] + 1
        st = [[0] * m for _ in range(K)]
        for i in range(m):
            st[0][i] = block_mins[i]
        for k in range(1, K):
            half = 1 << (k - 1)
            for i in range(m - (1 << k) + 1):
                lidx = st[k - 1][i]
                ridx = st[k - 1][i + half]
                st[k][i] = lidx if D[lidx] <= D[ridx] else ridx

        # 3) per-block ±1 signatures and precomputed tables
        block_sig = [None] * m
        block_tables = {}
        for bi in range(m):
            l = bi * b
            r = min(l + b, n)
            sig = 0
            for j in range(l, r - 1):
                sig = (sig << 1) | (1 if D[j + 1] > D[j] else 0)
            length = r - l
            key = (sig, length)
            block_sig[bi] = key
            if key not in block_tables:
                tbl = [[0] * length for _ in range(length)]
                for i in range(length):
                    tbl[i][i] = i
                    mpos = i
                    for j in range(i + 1, length):
                        if D[l + j] < D[l + mpos]:
                            mpos = j
                        tbl[i][j] = mpos
                block_tables[key] = tbl

        self._log = log
        self._st = st
        self._block_sig = block_sig
        self._block_tables = block_tables

    def query(self, l, r):
        """
        Return index k in [l..r] minimising D[k], in O(1).
        """
        b, n = self.b, self.n
        bi, bj = l // b, r // b
        if bi == bj:
            tbl = self._block_tables[self._block_sig[bi]]
            off = bi * b
            return off + tbl[l - off][r - off]

        # left block
        le = (bi + 1) * b - 1
        tbl = self._block_tables[self._block_sig[bi]]
        left_min = bi * b + tbl[l - bi * b][le - bi * b]

        # right block
        rs = bj * b
        tbl = self._block_tables[self._block_sig[bj]]
        right_min = bj * b + tbl[0][r - bj * b]

        # middle blocks
        mid_min = None
        if bi + 1 <= bj - 1:
            L, R = bi + 1, bj - 1
            length = R - L + 1
            k = self._log[length]
            i1 = self._st[k][L]
            i2 = self._st[k][R - (1 << k) + 1]
            mid_min = i1 if self.D[i1] <= self.D[i2] else i2

        best = left_min
        if self.D[right_min] < self.D[best]:
            best = right_min
        if mid_min is not None and self.D[mid_min] < self.D[best]:
            best = mid_min
        return best


# Globals for forward LCA
precomputed_parent_f = None
precomputed_depth_f = None
euler_E_f = None
euler_first_f = None
rmq_f = None
precomputed_preorder_f = None
precomputed_subtree_end_f = None
precomputed_preorder_nodes_f = None


def _build_forward_rooted_tree(graph, root):
    """Build the rooted tree induced by the allowed forward traversal."""

    parent = {root: root}
    depth = {root: 0}
    q = deque([root])
    while q:
        u = q.popleft()
        for v, data in graph[u].items():
            if v not in depth and allowed_edge(u, v, data.get("color")):
                parent[v] = u
                depth[v] = depth[u] + 1
                q.append(v)

    tree = defaultdict(list)
    for v, p in parent.items():
        if v != p:
            tree[p].append(v)
    return parent, depth, tree


def precompute_unique_paths_forward(graph, root, *, return_timings: bool = False):
    """
    1) BFS from root → parent[], depth[]
    2) Build tree adjacency, Euler-tour → E[], D[], first[]
    3) Build ±1-RMQ on D[]
    """
    global precomputed_parent_f, precomputed_depth_f
    global euler_E_f, euler_first_f, rmq_f
    global precomputed_preorder_f, precomputed_subtree_end_f, precomputed_preorder_nodes_f

    bfs_start = time.perf_counter()
    parent, depth, tree = _build_forward_rooted_tree(graph, root)
    rooted_bfs_depth_s = time.perf_counter() - bfs_start

    # Euler-tour DFS
    dfs_start = time.perf_counter()
    E = []
    D = []
    first = {}
    preorder = {}
    subtree_end = {}
    preorder_nodes = []
    sys.setrecursionlimit(10**7)

    def dfs(u):
        preorder[u] = len(preorder_nodes)
        preorder_nodes.append(u)
        first[u] = len(E)
        E.append(u)
        D.append(depth[u])
        for w in tree[u]:
            dfs(w)
            E.append(u)
            D.append(depth[u])
        subtree_end[u] = len(preorder_nodes) - 1

    dfs(root)
    preorder_subtree_dfs_s = time.perf_counter() - dfs_start

    # build RMQ
    rmq_start = time.perf_counter()
    rmq_f = RMQLinear(D)
    rmq_build_s = time.perf_counter() - rmq_start

    # store globals
    precomputed_parent_f = parent
    precomputed_depth_f = depth
    euler_E_f = E
    euler_first_f = first
    precomputed_preorder_f = preorder
    precomputed_subtree_end_f = subtree_end
    precomputed_preorder_nodes_f = preorder_nodes

    if return_timings:
        rooted_preorder_tables_s = rooted_bfs_depth_s + preorder_subtree_dfs_s
        lca_preorder_tables_s = rooted_preorder_tables_s + rmq_build_s
        return {
            "rooted_bfs_depth_s": rooted_bfs_depth_s,
            "preorder_subtree_dfs_s": preorder_subtree_dfs_s,
            "rmq_build_s": rmq_build_s,
            "rooted_preorder_tables_s": rooted_preorder_tables_s,
            "lca_preorder_tables_s": lca_preorder_tables_s,
        }


def find_lca_forward(node1, node2):
    """
    O(1) LCA lookup after precompute.
    """
    global euler_E_f, euler_first_f, rmq_f
    if euler_E_f is None:
        raise RuntimeError("Call precompute_unique_paths_forward(...) first")
    first = euler_first_f
    if node1 not in first or node2 not in first:
        return None
    i, j = first[node1], first[node2]
    if i > j:
        i, j = j, i
    idx = rmq_f.query(i, j)
    return euler_E_f[idx]
