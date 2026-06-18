"""分支推理内存共享 / Copy-on-Write（赛题优化方向之二）。

针对智能体多路径决策过程（如 Tree-of-Thought），设计 KV Cache 的共享与
Copy-on-Write 机制，避免分支带来的内存爆炸。详见 ``docs/技术方案.md`` §3.2。

与 vLLM APC 的区别
------------------
vLLM 的 Automatic Prefix Caching 会复用「完全相同的 token 前缀」的 KV——
对多分支的**公共前缀**已生效。但 APC 不感知「分支语义」：
- 不知道哪些序列属于同一棵推理树；
- 不知道某分支被「剪枝」了（只能靠 LRU 被动淘汰）；
- 不会主动回收「已无引用的分支独有 KV」。

Mimir 的 ``BranchTree`` 在 APC 之上提供分支语义：
1. **显式分支树**：记录每个分支的共享前缀长度与独有 token 区间。
2. **增量 KV 记账**：计算「逻辑 KV token 数」= 共享前缀（计一次）+ 各分支独有部分。
   对比「朴素 KV」= 每分支全量。CoW 节省 = 朴素 - 逻辑。
3. **剪枝回收**：分支被剪枝时立即回收其独有 KV（引用计数归零）。

本模块是纯数据结构 + 记账逻辑（不直接操作 vLLM KV，便于单测与报告）。
真实 KV 共享由 vLLM APC 在引擎层完成；本模块负责**度量、决策与回收信号**。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BranchNode:
    """分支树节点。

    ``shared_prefix_len``：与父节点共享的 token 数（公共前缀）。
    ``own_tokens``：本分支独有的 token（CoW：仅这部分新分配 KV）。
    ``pruned``：是否已剪枝（剪枝后独有 KV 应被回收）。
    """

    node_id: int
    parent_id: int | None
    shared_prefix_len: int
    own_tokens: int
    pruned: bool = False
    children: list[int] = field(default_factory=list)


@dataclass
class BranchTree:
    """分支推理树 + CoW KV 记账。"""

    root_prefix_len: int  # 所有分支共享的最底层前缀（如 system+user）
    nodes: dict[int, BranchNode] = field(default_factory=dict)
    _next_id: int = 0

    def __post_init__(self) -> None:
        if not self.nodes:
            # 根节点持有公共前缀（own_tokens = root_prefix_len），_total_len(root)=root_prefix_len
            root = BranchNode(
                node_id=0, parent_id=None, shared_prefix_len=0, own_tokens=self.root_prefix_len
            )
            self.nodes[0] = root
            self._next_id = 1

    def add_branch(self, parent_id: int, own_tokens: int) -> BranchNode:
        """在 parent 下新增一个分支，独有 ``own_tokens`` 个 token。

        共享前缀长度 = parent 的（共享 + 独有）累计 token 数。
        """
        parent = self.nodes[parent_id]
        parent_total = self._total_len(parent_id)
        node = BranchNode(
            node_id=self._next_id,
            parent_id=parent_id,
            shared_prefix_len=parent_total,
            own_tokens=own_tokens,
        )
        self.nodes[self._next_id] = node
        parent.children.append(self._next_id)
        self._next_id += 1
        return node

    def _total_len(self, node_id: int) -> int:
        """该节点的累计 token 长度（共享前缀 + 独有）。"""
        n = self.nodes[node_id]
        return n.shared_prefix_len + n.own_tokens

    def prune(self, node_id: int) -> int:
        """剪枝一个分支（及其子树），返回回收的「独有 KV token 数」。

        被剪枝分支的独有 token 不再被引用，应立即回收（CoW 引用计数归零）。
        """
        reclaimed = 0
        stack = [node_id]
        while stack:
            nid = stack.pop()
            n = self.nodes[nid]
            if n.pruned:
                continue
            n.pruned = True
            reclaimed += n.own_tokens  # 仅独有部分可回收（共享前缀仍被其他分支用）
            stack.extend(c for c in n.children if not self.nodes[c].pruned)
        return reclaimed

    # ---- 记账：朴素 vs CoW ----

    def naive_kv_tokens(self) -> int:
        """朴素 KV：每个活跃分支都存全量 token（无共享）。

        = Σ(活跃分支的累计长度)。分支爆炸时这是 O(N·L)。
        """
        total = 0
        for n in self.nodes.values():
            if not n.pruned and n.parent_id is not None:  # 跳过根（根无独立请求）
                total += self._total_len(n.node_id)
        return total

    def cow_kv_tokens(self) -> int:
        """CoW KV：公共前缀（根）只计一次 + 各非根分支独有部分。

        = 根 own_tokens（即 root_prefix_len，计一次）+ Σ(非根活跃分支 own_tokens)。
        非根分支的共享前缀部分不重复计入（被根覆盖）。
        """
        root = self.nodes[0]
        if root.pruned:
            return 0
        own_sum = sum(
            n.own_tokens for n in self.nodes.values() if not n.pruned and n.parent_id is not None
        )
        return root.own_tokens + own_sum

    def cow_savings(self) -> dict[str, int | float]:
        """CoW 相对朴素的节省。"""
        naive = self.naive_kv_tokens()
        cow = self.cow_kv_tokens()
        saved = max(0, naive - cow)
        pct = (saved / naive * 100) if naive else 0.0
        return {
            "naive_kv_tokens": naive,
            "cow_kv_tokens": cow,
            "saved_tokens": saved,
            "savings_pct": round(pct, 1),
            "active_branches": sum(
                1 for n in self.nodes.values() if not n.pruned and n.parent_id is not None
            ),
        }


def build_tot_tree(
    root_prefix_len: int,
    num_branches: int,
    own_tokens_per_branch: int,
    depth: int = 1,
) -> BranchTree:
    """构造一棵典型的 ToT 树：根下 ``num_branches`` 个分支，每分支 ``depth`` 层。

    每层每个活跃节点再分 ``num_branches`` 个子分支（模拟逐层展开）。
    """
    tree = BranchTree(root_prefix_len=root_prefix_len)
    frontier = [0]  # 根
    for d in range(depth):
        new_frontier = []
        for pid in frontier:
            for _ in range(num_branches):
                node = tree.add_branch(pid, own_tokens=own_tokens_per_branch)
                new_frontier.append(node.node_id)
        frontier = new_frontier
    return tree
