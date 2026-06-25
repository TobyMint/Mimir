# ruff: noqa: E501
"""验证 pin+SSC > pin(strong 档,8 agent × 10 轮,改后)。

修复后:pin TTL 真到期(取消 waiting 绕过)+ SSC 每轮 store + SSC reload 兜底。
预期:pin(TTL 到期,hit 降)< pin+SSC(SSC 兜底,hit 恢复)。
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_e2e_three_tier import run_side  # noqa: E402

results = {}
for mode in ["pin", "pinsc"]:
    print(f"\n=== {mode} (strong, 8×10) ===", flush=True)
    if mode == "pinsc":
        subprocess.run(["rm", "-rf", "/dev/shm/ssc_e2e"], check=False)
        os.makedirs("/dev/shm/ssc_e2e", exist_ok=True)
    res = run_side(3, mode, 8, 10, 2000, 12, 3000, 512, 0.55, 42)
    if res is None:
        print(f"  {mode}: FAILED", flush=True)
        continue
    print(f"  hit={res['hit_ratio']} ttft={res['mean_ttft_ms']}ms "
          f"prefill={res['total_new_prefill_tokens']} total={res['total_time_s']}s", flush=True)
    results[mode] = res

if "pin" in results and "pinsc" in results:
    p, s = results["pin"], results["pinsc"]
    print("\n=== 对比 ===", flush=True)
    print(f"  pin:     hit={p['hit_ratio']} prefill={p['total_new_prefill_tokens']}", flush=True)
    print(f"  pin+SSC: hit={s['hit_ratio']} prefill={s['total_new_prefill_tokens']}", flush=True)
    dh = (s['hit_ratio'] or 0) - (p['hit_ratio'] or 0)
    print(f"  pin+SSC - pin: hit +{dh:.3f}", flush=True)
