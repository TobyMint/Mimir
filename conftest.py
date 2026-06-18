"""确保仓库根目录在 sys.path 上，使 ``import benchmarks`` 在测试中可用。"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
