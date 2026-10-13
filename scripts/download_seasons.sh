#!/usr/bin/env bash
# Download specific seasons of the Robotic Origami Challenge dataset,
# LeRobot v3.0 branch only (skips the redundant lerobotv2.1 copy).
# TACTILE IS INCLUDED BY DEFAULT (both tactile_deform and tactile_raw).
#
# Prereqs (once): accept dataset terms while logged in on HF, then:
#   ~/.local/bin/hf auth login        # paste a read token
#
# Usage:
#   ./scripts/download_seasons.sh --first 3           # first 3 seasons, full (with tactile) ~45 GB
#   ./scripts/download_seasons.sh --first 3 --no-raw  # first 3, but drop tactile_raw ~12 GB
#   ./scripts/download_seasons.sh --random --budget-gb 200          # ~27 RANDOM seasons up to 200 GB (with tactile)
#   ./scripts/download_seasons.sh --random --budget-gb 200 --seed 7 # reproducible different draw
#   ./scripts/download_seasons.sh --random --budget-gb 200 --no-raw # ~61 random seasons, no tactile_raw
#   ./scripts/download_seasons.sh season_POC22032_2026_05_14_19_21_01_train season_POC22032_2026_05_14_20_40_58_train
#   ./scripts/download_seasons.sh --no-raw season_POC22061_2026_05_18_19_25_41_train
#
# Notes:
#   ~7.5 GB/season with tactile_raw (avg), ~3.3 GB/season without it.
#   --random reads outputs/data_analysis/season_sizes.json (run scripts/estimate_sizes.py once first),
#   excludes empty + already-downloaded seasons, and packs seasons up to the budget.
set -euo pipefail

REPO="SharpaIT/Robotic_Origami_Challenge"
HF="${HF:-$HOME/.local/bin/hf}"
# DEST is overridable via env (e.g. DEST=/media/robot/T7-Isma1/origami_no_lines_data/Robotic_Origami_Challenge
# to download onto an external drive); default = the repo's data/ dir, unchanged behavior.
DEST="${DEST:-$(cd "$(dirname "$0")/.." && pwd)/data/Robotic_Origami_Challenge}"
export HF_HUB_ENABLE_HF_TRANSFER=1

NO_RAW=0
FIRST_N=""
RANDOM_MODE=0
BUDGET_GB=200
SEED=42
SEASONS=()
HERE="$(cd "$(dirname "$0")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-raw) NO_RAW=1; shift ;;
    --first)  FIRST_N="$2"; shift 2 ;;
    --random) RANDOM_MODE=1; shift ;;
    --budget-gb) BUDGET_GB="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    season_*) SEASONS+=("$1"); shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

PY="${PY:-python}"

# If --first N was given, pull the first N season names from the live listing.
if [[ -n "$FIRST_N" ]]; then
  mapfile -t ALL < <("$HERE/list_seasons.sh")
  SEASONS=("${ALL[@]:0:$FIRST_N}")
fi

# If --random: let pick_seasons.py choose a budget-bounded random set.
if [[ $RANDOM_MODE -eq 1 ]]; then
  PICK_ARGS=(--budget-gb "$BUDGET_GB" --seed "$SEED")
  [[ $NO_RAW -eq 1 ]] && PICK_ARGS+=(--no-raw)
  mapfile -t SEASONS < <("$PY" "$HERE/pick_seasons.py" "${PICK_ARGS[@]}")
fi

if [[ ${#SEASONS[@]} -eq 0 ]]; then
  echo "No seasons specified. Use --first N or pass season_* names." >&2
  echo "List them with: ./scripts/list_seasons.sh" >&2
  exit 1
fi

EXCLUDE=()
[[ $NO_RAW -eq 1 ]] && EXCLUDE=(--exclude "*/tactile_raw/*")
[[ $NO_RAW -eq 1 ]] && echo ">> tactile_raw EXCLUDED" || echo ">> tactile_raw INCLUDED"

for S in "${SEASONS[@]}"; do
  echo "==> $S"
  "$HF" download "$REPO" \
    --repo-type dataset \
    --include "${S}/lerobot3.0/*" \
    "${EXCLUDE[@]}" \
    --local-dir "$DEST"
done

echo ">> Done. Downloaded ${#SEASONS[@]} season(s) to: $DEST"
du -sh "$DEST" 2>/dev/null || true
