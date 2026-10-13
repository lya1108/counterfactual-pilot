#!/usr/bin/env bash
# Read-only status of the online collection and the learning curve. Touches nothing.
cd ~/yuna_ws/xattn_detector
L=results/online_collect.log

# `pgrep -f` also matches any shell whose command line merely contains the pattern, so check that the
# process's own executable is python rather than that the string appears somewhere in its arguments.
running() { ps -eo args= 2>/dev/null | awk -v p="$1" '$1 ~ /python/ && index($0, "online_runner.py " p) {f=1} END{exit !f}'; }

last=$(grep -oP '^\[\K[0-9]+(?=/)' "$L" 2>/dev/null | tail -1); last=${last:-0}
tot=$(grep -oP '^\[[0-9]+/\K[0-9]+' "$L" 2>/dev/null | tail -1); tot=${tot:-234}
running collect && st="수집 진행중" || st="수집 종료"
pct=$(( last * 100 / tot ))
bar=$(printf '%*s' $(( pct / 4 )) '' | tr ' ' '#')
printf 'SAFE online rollout   %s\n\n  진행  %d / %d  [%-25s] %d%%\n' "$st" "$last" "$tot" "$bar" "$pct"

s=$(grep -cP '^\[[0-9]+/[0-9]+\].* S  calls=' "$L" 2>/dev/null); s=${s:-0}
f=$(grep -cP '^\[[0-9]+/[0-9]+\].* F  calls=' "$L" 2>/dev/null); f=${f:-0}
[ $((s+f)) -gt 0 ] && printf '  결과  성공 %d / 실패 %d  (성공률 %d%%)\n' "$s" "$f" $(( s*100/(s+f) ))
tail -1 "$L" 2>/dev/null | sed 's/^/  최근  /'
echo
if running curve; then
  echo "  학습곡선 진행중"
elif [ -f results/online_curve_main.json ]; then
  echo "  학습곡선 완료 -> results/online_curve_main.json"
else
  echo "  학습곡선 대기중 (수집 종료 후 자동 시작)"
fi
