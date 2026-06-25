# ruff: noqa: E501
"""诊断 pin TTL 是否真到期:跑 strong 干扰 pin 小规模,看 PIN_EXPIRE 日志。

若 PIN_EXPIRE 出现 → pin 在 gap 期间到期释放(那 SSC 该能兜底,Inject=0 是 SSC bug)。
若无 PIN_EXPIRE → pin 没到期(保活),SSC 没空间兜底(需缩短 TTL 或加长 gap)。
"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_e2e_three_tier import CHILD  # noqa: E402

env = dict(os.environ)
env["MIMIR_DEBUG_PIN"] = "1"
env["VLLM_LOGGING_LEVEL"] = "WARNING"
# 2 agent × 3 round, strong 干扰(12×3000+512), pin, util 0.55
r = subprocess.run(
    [sys.executable, "-c", CHILD, "3", "pin", "2", "3", "2000", "12", "3000", "512",
     "0.55", "42"],
    env=env, timeout=600)
print("rc", r.returncode)
