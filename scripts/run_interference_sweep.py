# ruff: noqa: E501
"""干扰强度扫描:反驳"native 0% 是刻意打地板"的质疑。

固定 util/agents/rounds/ctx,扫 4 档干扰压力,看:
  - native 命中率随干扰压力的退化曲线(无压→高命中,证明没打地板;高压→0%,证明 pin 在压力下的价值)
  - pin / pin+SSC 在各档是否稳定领先(pin 保活不受 LRU 干扰;SSC 兜底补 pin 保不住的)

复用 run_e2e_three_tier.CHILD / run_side(subprocess 隔离,add_request+step 交错)。
none 档只跑 native(证明 native 在无压力下本可高命中,不是被我们写坏);其余 3 档跑 native/pin/pinsc。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_e2e_three_tier import run_side  # noqa: E402  (复用 subprocess 隔离 + CHILD)

# 4 档干扰强度(固定 util=0.55 → KV 池 ~28K token;agents=8 × ctx=2000 → agent KV ~16K)
# none   : 无干扰           → native 应高命中(天花板,证明 native 没被写坏)
# weak   : ~5K token 干扰   → native 部分命中
# medium : ~14K token 干扰  → native 命中退化
# strong : ~42K token 干扰  → native 0%(原 e2e 场景,池被挤满)
PROFILES = [
    {"name": "none",   "intf_n": 0,  "intf_ctx": 1500, "intf_maxtok": 128, "modes": ["native"]},
    {"name": "weak",   "intf_n": 3,  "intf_ctx": 1500, "intf_maxtok": 128, "modes": ["native", "pin", "pinsc"]},
    {"name": "medium", "intf_n": 6,  "intf_ctx": 2000, "intf_maxtok": 256, "modes": ["native", "pin", "pinsc"]},
    {"name": "strong", "intf_n": 12, "intf_ctx": 3000, "intf_maxtok": 512, "modes": ["native", "pin", "pinsc"]},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--agents", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--ctx", type=int, default=2000)
    ap.add_argument("--util", type=float, default=0.55)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="benchmark_results")
    args = ap.parse_args()

    print(f"GPU {args.gpu}, {args.agents} agents × {args.rounds} rounds, util={args.util}, ctx={args.ctx}", flush=True)
    print("扫干扰: none/weak/medium/strong × native/pin/pinsc (none 只 native)", flush=True)

    sweep = {"load": vars(args), "profiles": {}}
    for prof in PROFILES:
        pname = prof["name"]
        print(f"\n##### 干扰档: {pname}  intf_n={prof['intf_n']} ctx={prof['intf_ctx']} "
              f"maxtok={prof['intf_maxtok']} #####", flush=True)
        sweep["profiles"][pname] = {"config": {k: prof[k] for k in
                                               ("intf_n", "intf_ctx", "intf_maxtok")},
                                    "results": {}}
        for mode in prof["modes"]:
            print(f"  --- {pname}/{mode} ---", flush=True)
            # SSC store 在 pinsc 档前清,避免上档残留影响前缀匹配
            if mode in ("ssc", "pinsc"):
                subprocess.run(["rm", "-rf", "/dev/shm/ssc_e2e"], check=False)
                os.makedirs("/dev/shm/ssc_e2e", exist_ok=True)
            res = run_side(args.gpu, mode, args.agents, args.rounds, args.ctx,
                           prof["intf_n"], prof["intf_ctx"], prof["intf_maxtok"],
                           args.util, args.seed)
            if res is None:
                print(f"    {pname}/{mode}: FAILED", flush=True)
                continue
            print(f"    total={res['total_time_s']}s ttft={res['mean_ttft_ms']}ms "
                  f"hit={res['hit_ratio']} prefill={res['total_new_prefill_tokens']} "
                  f"done={res['n_agents_done']}/{res['n_agents']}", flush=True)
            sweep["profiles"][pname]["results"][mode] = res

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)
    path = out / "interference_sweep.json"
    path.write_text(json.dumps(sweep, indent=2, ensure_ascii=False), encoding="utf-8")

    # 总表:命中率随压力退化 + pin/pinsc 各档稳定优势
    print("\n\n========== 干扰强度扫描总表 ==========", flush=True)
    print(f"{'档':<8} {'mode':<8} {'total_s':>9} {'ttft_ms':>9} {'hit':>7} {'prefill':>9} {'done':>6}", flush=True)
    for prof in PROFILES:
        pname = prof["name"]
        for mode in prof["modes"]:
            r = sweep["profiles"][pname]["results"].get(mode)
            if r:
                print(f"{pname:<8} {mode:<8} {r['total_time_s']:>9} "
                      f"{(r['mean_ttft_ms'] if r['mean_ttft_ms'] is not None else 0):>9} "
                      f"{(r['hit_ratio'] if r['hit_ratio'] is not None else 0):>7} "
                      f"{r['total_new_prefill_tokens']:>9} "
                      f"{r['n_agents_done']:>3}/{r['n_agents']}", flush=True)

    # native 命中率退化曲线(核心论据)
    print("\n--- native 命中率随干扰压力退化 ---", flush=True)
    for prof in PROFILES:
        pname = prof["name"]
        r = sweep["profiles"][pname]["results"].get("native")
        hit = r["hit_ratio"] if r and r["hit_ratio"] is not None else "N/A"
        print(f"  {pname:<8}: native hit = {hit}", flush=True)

    print(f"\nJSON → {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
