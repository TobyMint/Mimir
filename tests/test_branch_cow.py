"""``mimir.branch.cow`` 单元测试（纯逻辑）。"""

from __future__ import annotations

from mimir.branch.cow import BranchTree, build_tot_tree


def test_single_layer_sharing() -> None:
    """4 分支共享 200 token 前缀，每分支独有 50：朴素=4*250，CoW=200+4*50。"""
    tree = build_tot_tree(root_prefix_len=200, num_branches=4, own_tokens_per_branch=50, depth=1)
    s = tree.cow_savings()
    assert s["active_branches"] == 4
    assert s["naive_kv_tokens"] == 4 * 250  # 1000
    assert s["cow_kv_tokens"] == 200 + 4 * 50  # 400
    assert s["saved_tokens"] == 600
    assert s["savings_pct"] == 60.0


def test_two_layer_tree() -> None:
    """depth=2, 2 branches/layer, 30 own each, root 100。

    L1: 2 nodes, each own 30, total 130
    L2: 4 nodes, each own 30, total (parent 130 + 30) = 160
    朴素 = 2*130 + 4*160 = 260 + 640 = 900
    CoW = 100 + (2+4)*30 = 100 + 180 = 280
    """
    tree = build_tot_tree(root_prefix_len=100, num_branches=2, own_tokens_per_branch=30, depth=2)
    s = tree.cow_savings()
    assert s["active_branches"] == 6
    assert s["naive_kv_tokens"] == 2 * 130 + 4 * 160
    assert s["cow_kv_tokens"] == 100 + 6 * 30


def test_prune_reclaims_own_tokens() -> None:
    tree = build_tot_tree(root_prefix_len=200, num_branches=4, own_tokens_per_branch=50, depth=1)
    # 剪掉一个分支（own 50）
    reclaimed = tree.prune(1)
    assert reclaimed == 50
    s = tree.cow_savings()
    assert s["active_branches"] == 3
    # CoW 现在少了一个分支的 50 own
    assert s["cow_kv_tokens"] == 200 + 3 * 50


def test_prune_subtree_reclaims_all_own() -> None:
    tree = build_tot_tree(root_prefix_len=100, num_branches=2, own_tokens_per_branch=30, depth=2)
    # 剪掉 L1 节点 1 及其 2 个 L2 子节点
    reclaimed = tree.prune(1)
    assert reclaimed == 30 + 2 * 30  # 自身 + 2 子
    s = tree.cow_savings()
    assert s["active_branches"] == 3  # 1 L1 + 2 L2


def test_more_branches_more_savings() -> None:
    """分支越多，CoW 相对朴素节省越大。"""
    s4 = build_tot_tree(200, 4, 50, 1).cow_savings()
    s8 = build_tot_tree(200, 8, 50, 1).cow_savings()
    assert s8["savings_pct"] > s4["savings_pct"]


def test_cow_never_exceeds_naive() -> None:
    for nb in (2, 4, 8):
        for d in (1, 2, 3):
            tree = build_tot_tree(150, nb, 40, d)
            s = tree.cow_savings()
            assert s["cow_kv_tokens"] <= s["naive_kv_tokens"]
