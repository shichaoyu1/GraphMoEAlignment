#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
pretrain=${IDEA3_PRETRAIN_ROOT:-"$repo/output/idea3_pareto_pretrain"}
output=${IDEA3_PARETO_OUTPUT:-"$repo/output/idea3_pareto_mature_forks"}
seed=${IDEA3_SEED:-42}
for task in common sensors; do
  initial="$pretrain/controlled/$task/moe_sft/seed$seed/best.pt"
  [[ -f "$initial" ]] || { echo "Missing supervised checkpoint: $initial" >&2; exit 1; }
done
mkdir -p "$output/logs"
exec 8>"$output/.forks.lock"
flock -n 8 || { echo 'A fork comparison is already using this output directory' >&2; exit 1; }
for task in common sensors; do
  initial="$pretrain/controlled/$task/moe_sft/seed$seed/best.pt"
  for branch in sft_top8 sft_best pg nft; do
    teacher=top8
    case "$branch" in
      sft_top8) method=moe_sft ;;
      sft_best) method=moe_sft; teacher=best ;;
      pg) method=moe_pg ;;
      nft) method=moe_nft ;;
    esac
    IDEA3_PARETO_OUTPUT="$output/$branch" IDEA3_TASK="$task" IDEA3_METHOD="$method" \
      bash "$repo/scripts/run_idea3_pareto_autodl.sh" controlled \
        --seed "$seed" --init-policy "$initial" --warmup 0 --updates 1000 \
        --teacher-mode "$teacher" --validate-every 100 \
        --eval-repeats 128 --test-repeats 256 "$@"
  done
done
