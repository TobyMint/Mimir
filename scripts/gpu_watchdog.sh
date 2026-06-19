#!/usr/bin/env bash
# GPU 监控：每 30 秒检查一次，发现 >=6GiB 空闲的卡就记录到 benchmark_results/gpu_log.txt
# 用法：nohup bash scripts/gpu_watchdog.sh &
LOG="benchmark_results/gpu_log.txt"
echo "=== GPU watchdog started $(date) ===" >> "$LOG"
while true; do
    TS=$(date '+%Y-%m-%d %H:%M:%S')
    LINE=$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    FREE_GPUTWO=$(echo "$LINE" | awk -F', ' '{print $2}' | sort -rn | head -1)
    if [ -n "$FREE_GPUTWO" ] && [ "$FREE_GPUTWO" -gt 6000 ]; then
        echo "$TS FREE=${FREE_GPUTWO}MiB >>> GPU IDLE, can run heavy benchmark" >> "$LOG"
    else
        echo "$TS max_free=${FREE_GPUTWO}MiB (busy)" >> "$LOG"
    fi
    sleep 30
done
