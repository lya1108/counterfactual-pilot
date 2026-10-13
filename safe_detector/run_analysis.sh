#!/usr/bin/env bash
# Train every detector / baseline cell on the collected LIBERO rollouts, then print one table.
#
# Two conditions are run for each cell:
#   full      whole episodes. Successes stop when solved, failures run to the timeout, so the call index alone
#             almost separates the classes -- read the `time` baseline row before reading anything else here.
#   prefix8   every episode truncated to its first 8 calls (80 environment steps). Every episode now has the same
#             index sequence, so `time` is exactly chance and the number means: is failure readable this early?
#
# Two splits: `init` holds out rollouts of tasks that were trained on; `task` holds out whole tasks.
set -u
PY=~/miniconda3/envs/fasi/bin/python
cd "$(dirname "$0")/.."
K=8

run () {  # run <tag> <extra args...>
  tag=$1; shift
  echo "---- $tag"
  # --data is resolved against safe_detector/ by train_detector.py, so it stays bare here
  $PY -u safe_detector/train_detector.py --data data --tag "$tag" "$@" 2>&1 \
    | grep -vE "^  epoch"
}

for split in init task; do
  for cond in full prefix8; do
    [ "$cond" = full ] && trunc="" || trunc="--max-calls $K"
    s=${split}_${cond}
    run "time_$s"            --model time   --split "$split" $trunc
    run "random_$s"          --model random --split "$split" $trunc
    for feat in vlm_mean act_mean; do
      run "logreg_${feat}_$s" --model logreg --feature "$feat" --split "$split" $trunc
      run "lstm_${feat}_$s"   --model lstm   --feature "$feat" --split "$split" $trunc
    done
    run "lstm_concat_$s"     --model lstm   --feature concat --split "$split" $trunc
  done
done

echo
$PY -u safe_detector/summarize.py
