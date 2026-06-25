# ruff: noqa: E501
"""补跑 pin+SSC(pinsc)4 档曲线,完成三步走 native→pin→pin+SSC。

修复后(pin TTL 真到期 + SSC 每轮 store + SSC reload 兜底),pinsc 在 medium/strong
该 > pin(SSC 兜底 pin 到期的);none/weak pin 不到期,SSC 不兜底(≈ pin 或略低)。
merge 进 interference_sweep.json,再重画图。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_e2e_three_tier import run_side  # noqa: E402

PROFILES = [
    {"name": "none", "intf_n": 0, "intf_ctx": 1500, "intf_maxtok": 128},
    {"name": "weak", "intf_n": 3, "intf_ctx": 1500, "intf_maxtok": 128},
    {"name": "medium", "intf_n": 6, "intf_ctx": 2000, "intf_maxtok": 256},
    {"name": "strong", "intf_n": 12, "intf_ctx": 3000, "intf_maxtok": 512},
]


def main():
    json_path = Path("benchmark_results/interference_sweep.json")
    sweep = json.loads(json_path.read_text(encoding="utf-8"))
    for prof in PROFILES:
        p = prof["name"]
        print(f"\n##### pinsc 补跑: {p} (intf_n={prof['intf_n']}) #####", flush=True)
        subprocess.run(["rm", "-rf", "/dev/shm/ssc_e2e"], check=False)
        os.makedirs("/dev/shm/ssc_e2e", exist_ok=True)
        res = run_side(3, "pinsc", 8, 10, 2000, prof["intf_n"], prof["intf_ctx"],
                       prof["intf_maxtok"], 0.55, 42)
        if res is None:
            print(f"  {p}/pinsc FAILED", flush=True)
            continue
        print(f"  hit={res['hit_ratio']} ttft={res['mean_ttft_ms']}ms "
              f"prefill={res['total_new_prefill_tokens']} total={res['total_time_s']}s "
              f"done={res['n_agents_done']}/{res['n_agents']}", flush=True)
        sweep["profiles"][p]["results"]["pinsc"] = res
        json_path.write_text(json.dumps(sweep, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    print(f"\nmerged → {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
