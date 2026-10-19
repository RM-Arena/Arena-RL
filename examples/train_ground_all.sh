#!/bin/sh
# Reproduce checkpoints/ground on one machine: universal ground policies, heavy-only
# overrides, the aerial dive policy, then per-model evaluation.
# Usage: sh examples/train_ground_all.sh runs/ground-v1 [workers] [device]
# DEVICE: auto (CUDA if present, else CPU), cpu, mps, cuda. On Apple silicon keep cpu/auto:
# rollout workers own the simulation cores and the small PPO update is fastest on CPU.
# Two families train at once so one job's idle cores are used by the other.
# Finished stages (metadata.json / evaluation.json present) are skipped, so re-running resumes.
set -u
OUT=${1:-runs/ground-v1}
WORKERS=${2:-8}
DEVICE=${3:-auto}
LIGHT="3 4 7 301 302 303 304 305 2731 2732 2771 2772"
HEAVY="2711 2712 2713"
mkdir -p "$OUT"
tg() { name=$1; shift; [ -f "$OUT/$name/metadata.json" ] || uv run arena-rl train-ground --workers "$WORKERS" --device "$DEVICE" --bc-epochs 30 --output "$OUT/$name" "$@" > "$OUT/$name.log" 2>&1; }
ev() { name=$1; shift; [ -f "$OUT/$name/evaluation.json" ] || uv run arena-rl evaluate-ground --checkpoint "$OUT/$name" --episodes-per-model 8 --workers "$WORKERS" --output "$OUT/$name/evaluation.json" "$@" > "$OUT/$name-eval.log" 2>&1; }
(
  tg navigation --family navigation --steps 1228800 --seed 601 --expert-episodes 320
  tg combat --family combat --steps 307200 --seed 602 --models $LIGHT --expert-episodes 320
  tg combat-heavy --family combat --steps 122880 --seed 612 --models $HEAVY --expert-episodes 160
) &
(
  tg defense --family defense --steps 409600 --seed 603 --models $LIGHT --expert-episodes 320
  tg defense-heavy --family defense --steps 40960 --seed 613 --models $HEAVY --expert-episodes 160
  tg support --family support --steps 204800 --seed 604 --expert-episodes 320
  [ -f "$OUT/aerial/metadata.json" ] || uv run arena-rl train-aerial --steps 153600 --envs "$WORKERS" --device "$DEVICE" --output "$OUT/aerial" > "$OUT/aerial.log" 2>&1
) &
wait
(ev navigation; ev combat --models $LIGHT; ev combat-heavy --models $HEAVY) &
(ev defense --models $LIGHT; ev defense-heavy --models $HEAVY; ev support
 [ -f "$OUT/aerial/evaluation.json" ] || uv run arena-rl evaluate-aerial --checkpoint "$OUT/aerial" --episodes 36 --output "$OUT/aerial/evaluation.json" > "$OUT/aerial-eval.log" 2>&1) &
wait
echo "trained and evaluated under $OUT; package with: uv run python examples/package_ground.py $OUT checkpoints/ground"
