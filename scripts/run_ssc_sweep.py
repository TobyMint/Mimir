# ruff: noqa: E501
"""补跑 native+SSC(ssc 档)曲线,完成三步走 native→pin→native+SSC。

之前 sweep 跑的是 pinsc(pin+SSC),诊断显示 pin+SSC 冲突(SSC load 在 pin 工作时
完全不触发——Inject KV=0)。用户三步走第三步是 native+SSC(SSC 代替 pin,非 pin+SSC),
故补跑 ssc 档 weak/medium/strong,merge 进 interference_sweep.json,再重画图。

ssc 档每跑前清 /dev/shm/ssc_e2e(SSC store 从零起步)。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_e2e_three_tier import run_side  # noqa: E402

PROFILES = [
    {"name": "weak", "intf_n": 3, "intf_ctx": 1500, "intf_maxtok": 128},
    {"name": "medium", "intf_n": 6, "intf_ctx": 2000, "intf_maxtok": 256},
    {"name": "strong", "intf_n": 12, "intf_ctx": 3000, "intf_maxtok": 512},
]


def main():
    json_path = Path("benchmark_results/interference_sweep.json")
    sweep = json.loads(json_path.read_text(encoding="utf-8"))
    for prof in PROFILES:
        p = prof["name"]
        print(f"\n##### ssc 补跑: {p} (intf_n={prof['intf_n']}) #####", flush=True)
        subprocess.run(["rm", "-rf", "/dev/shm/ssc_e2e"], check=False)
        os.makedirs("/dev/shm/ssc_e2e", exist_ok=True)
        res = run_side(3, "ssc", 8, 10, 2000, prof["intf_n"], prof["intf_ctx"],
                       prof["intf_maxtok"], 0.55, 42)
        if res is None:
            print(f"  {p}/ssc FAILED", flush=True)
            continue
        print(f"  total={res['total_time_s']}s ttft={res['mean_ttft_ms']}ms "
              f"hit={res['hit_ratio']} prefill={res['total_new_prefill_tokens']} "
              f"done={res['n_agents_done']}/{res['n_agents']}", flush=True)
        sweep["profiles"][p]["results"]["ssc"] = res
        # 增量写盘(每档跑完即存,防中断丢)
        json_path.write_text(json.dumps(sweep, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    print(f"\nmerged → {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
