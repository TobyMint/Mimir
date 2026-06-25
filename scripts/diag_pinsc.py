# ruff: noqa: E501
"""诊断 pin+SSC:为什么 sweep strong/pinsc 的 SSC reload 没贡献命中(77.6%≈pin)。

假设:pin 保活让 agent KV 留 GPU → 下轮 APC num_computed_tokens 已高 →
SSC get_num_new_matched_tokens 返回 (matched - computed) = 0 → SSC load 不触发,
只剩 store 开销(故 strong/pinsc 988s)。

小规模(2 agent × 3 round, strong 干扰)跑 pinsc,stderr 流出看 SSC 日志:
  - "External Cache Hit"  = SSC 前缀匹配命中(说明 load 该触发)
  - "Inject KV cache"     = SSC load 真注入了 KV
  - "SSC load skip"       = load 容错跳过(文件不存在/形状不匹配)
  - 缺这几行 = SSC load 根本没进入 → 证实假设(pin 保活让 load 不触发)
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
subprocess.run(["rm", "-rf", "/dev/shm/ssc_e2e"], check=False)
os.makedirs("/dev/shm/ssc_e2e", exist_ok=True)
from run_e2e_three_tier import CHILD  # noqa: E402

env = dict(os.environ)
env["VLLM_LOGGING_LEVEL"] = "INFO"
# 2 agent × 3 round, strong 干扰(8×3000+512), pinsc, util 0.55, seed 42
r = subprocess.run(
    [sys.executable, "-c", CHILD, "3", "pinsc", "2", "3", "2000", "8", "3000", "512",
     "0.55", "42"],
    env=env, timeout=900)
print("rc", r.returncode)
