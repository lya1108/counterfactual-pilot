#!/usr/bin/env bash
# Collect SAFE rollouts for every LIBERO suite, one suite per process (a crash in one does not lose the others).
set -u
PY=~/miniconda3/envs/fasi/bin/python
cd ~/yuna_ws
for suite in libero_spatial libero_object libero_goal libero_10; do
  echo "######## $suite ########"
  $PY -u safe_detector/collect_rollouts.py --suite "$suite" --tasks 0-9 --inits 0-19 \
      --out "data/$suite" 2>&1 | grep -vE "EGL|OpenGL|binding_utils|egl_context|Exception ignored|^\s+(self|EGL|raise|<OpenGL|\)|result|err|cArguments|baseOperation)"
done
echo "######## ALL DONE ########"
