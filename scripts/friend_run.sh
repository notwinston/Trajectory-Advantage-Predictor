#!/usr/bin/env bash
# ============================================================================
# TAP label generator — run on YOUR Prime Intellect account to mint labels.
#
# It spins up a GPU pod on your account and runs the accuracy-lift battery
# (Qwen3-1.7B) in QUALITY mode: every cohort is measured with 3 training seeds
# and a 128-item greedy probe, so each lift label is denoised (not just
# plentiful). Labels download to ./outputs/, the pod terminates, and it repeats
# with fresh cohorts. Quality > quantity — fewer labels, but clean ones.
# Leave it running; Ctrl-C stops.
#
# ---- one-time setup (you handle this) -------------------------------------
#   pip install prime-cli            # or: uv tool install prime-cli
#   prime login                      # your own account
#   ssh-keygen -t ed25519            # if you don't already have ~/.ssh/id_ed25519
#   prime config set-ssh-key-path ~/.ssh/id_ed25519
#   git clone https://github.com/notwinston/Inference_Time.git
#   cd Inference_Time && git checkout ajain/v3
# (needs python3 + ssh + rsync on your machine — no other pip installs.)
#
# ---- usage ----------------------------------------------------------------
#   scripts/friend_run.sh <your_name> [domain] [gpu_count]
#     domain    = math | code | science | mmlu     (default: math)
#     gpu_count = GPUs per pod (default 1; N => N parallel shards => ~Nx faster).
#                 Quality mode is ~6x heavier per label, so pass 2-8 if you can.
#
#   examples:
#     scripts/friend_run.sh alice math
#     scripts/friend_run.sh bob   code 2
#
# When you're done, send me the whole  outputs/friend_<your_name>_<domain>/  folder.
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

NAME="${1:?usage: scripts/friend_run.sh <your_name> [domain] [gpu_count]}"
DOMAIN="${2:-math}"
GC="${3:-1}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
PROVIDER="${PROVIDER:-datacrunch}"

case "$DOMAIN" in
  math|code)    MAXTOK=768; MB=2; EB=12 ;;  # long gen: small GRAD batch (OOM) + big eval batch (speed)
  science|mmlu) MAXTOK=256; MB=8; EB=8 ;;
  *) echo "domain must be one of: math code science mmlu"; exit 1 ;;
esac

# GPU types tried in order until one has an available offer on your account.
GPU_TYPES=(A100_80GB A100_40GB H100_80GB)

echo "### TAP labels | name=$NAME domain=$DOMAIN gpus=$GC provider=$PROVIDER"
echo "### labels will land in outputs/friend_${NAME}_${DOMAIN}/  (Ctrl-C to stop)"

SEED="${START_SEED:-0}"
while true; do
  OUT="outputs/friend_${NAME}_${DOMAIN}/seed_${SEED}"
  launched=0
  for GT in "${GPU_TYPES[@]}"; do
    echo "=== [$DOMAIN seed=$SEED] trying ${GC}x ${GT} on ${PROVIDER} ==="
    if python3 run_tap_pod.py \
        --provider "$PROVIDER" --gpu-type "$GT" --gpu-count "$GC" --shard-total "$GC" \
        --domain "$DOMAIN" --model-name Qwen/Qwen3-1.7B \
        --cohort-size 4 --group-size 8 --temperature 1.2 --max-new-tokens "$MAXTOK" --micro-batch "$MB" --eval-batch "$EB" \
        --grpo-steps 15 --lr 3e-4 --probe-size 128 --probe-k 4 --anchors-per-chain 1 \
        --n-random 24 --seeds 3 --seed "$SEED" \
        --ssh-key "$SSH_KEY" --output-dir "$OUT"; then
      n=$(cat "$OUT"/labels.jsonl 2>/dev/null | wc -l | tr -d ' ')
      echo "=== batch done: ~${n} labels saved to ${OUT} ==="
      launched=1
      break
    fi
    echo "=== ${GC}x ${GT} unavailable/failed, trying next type ==="
  done
  if [ "$launched" -eq 0 ]; then
    echo "### no offers available right now — waiting 120s, then retrying ..."
    sleep 120
    continue
  fi
  SEED=$((SEED + 1))
done
