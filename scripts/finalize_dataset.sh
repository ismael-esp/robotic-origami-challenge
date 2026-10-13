#!/usr/bin/env bash
# Run once the dataset download is COMPLETE. Verifies integrity, attaches the
# pre/post-cutoff regime discriminator, runs the full analysis (overall + the
# post-cutoff regime the user will train on), and rebuilds the HTML reports.
#
# Usage: ./scripts/finalize_dataset.sh   (uses the origami conda env)
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"

echo "############ 1/6  verify data integrity ############"
$PY scripts/verify_data.py

echo; echo "############ 2/6  regime labels (pre/post cutoff) ############"
$PY scripts/label_regime.py

echo; echo "############ 3/6  analysis — ALL seasons ############"
$PY scripts/analyze_seasons.py --regime all  2>/dev/null | tail -n 20

echo; echo "############ 4/6  analysis — POST-cutoff (training target) ############"
$PY scripts/analyze_seasons.py --regime post 2>/dev/null | tail -n 20

echo; echo "############ 5/6  HTML report — ALL ############"
$PY scripts/build_report.py --regime all  2>/dev/null

echo; echo "############ 6/6  HTML report — POST-cutoff ############"
$PY scripts/build_report.py --regime post 2>/dev/null

echo; echo "############ DONE ############"
echo "Reports:   outputs/reports/data_report.html  |  outputs/reports/data_report_post.html"
echo "Analysis:  docs/data_analysis.md     |  docs/data_analysis_post.md"
echo "Regime:    outputs/regime_labels.{json,csv}"
