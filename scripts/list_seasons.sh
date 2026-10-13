#!/usr/bin/env bash
# List all season directories in the dataset (no auth needed for the listing).
set -euo pipefail
REPO="SharpaIT/Robotic_Origami_Challenge"
curl -s "https://huggingface.co/api/datasets/${REPO}/tree/main?recursive=false" \
  | python3 -c 'import sys,json;[print(e["path"]) for e in json.load(sys.stdin) if e["type"]=="directory" and e["path"].startswith("season_")]'
