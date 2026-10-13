#!/usr/bin/env bash
# After collection: re-collect the flagged episodes with the correct warm-up, then test whether any remaining
# flagged episode still reaches the outcome its stored label claims (with a clean-episode control group).
set -u
PY=~/miniconda3/envs/fasi/bin/python
cd ~/yuna_ws
echo "######## REFIX ########"
$PY -u xattn_detector/collect_tokens.py --out data/tokens --refix 2>&1 \
  | grep -vE "EGL|OpenGL|binding_utils|egl_context|Exception ignored|^\s+(self|EGL|raise|<OpenGL|\)|result|err|cArguments|baseOperation)"
echo "######## LABEL CHECK ########"
$PY -u xattn_detector/label_check.py --tokens data/tokens --runs 2 --control 20 2>&1 \
  | grep -vE "EGL|OpenGL|binding_utils|egl_context|Exception ignored|^\s+(self|EGL|raise|<OpenGL|\)|result|err|cArguments|baseOperation)"
echo "######## VERIFY DONE ########"
