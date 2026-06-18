"""分支推理内存共享（赛题优化方向之二）。

针对智能体多路径决策过程（如 Tree-of-Thought），设计 KV Cache 的共享与
Copy-on-Write 机制，避免分支带来的内存爆炸。

计划实现：
    - Tree-based KV：将分支推理组织为树结构，共享公共前缀块
    - Copy-on-Write：仅当分支真正修改对应 token 的 KV 时物理拷贝
    - 分支裁剪与回收

详见 ``docs/技术方案.md`` §3.2。
"""
