# outputs/

Generated artifacts. Large files (checkpoints, Docker archives, frame caches, videos) are
not tracked in git; only the small files the code and the results depend on are.

| entry | what | in git |
|---|---|---|
| `eval/*.json` | gate-battery, visual-grounding, closed-loop-drift and lock-in results per checkpoint (`gates_*`, `grounding_*`, `closed_loop_*`, `lockin_*`), plus diagnostics from the live-test analysis (`input_authority`, `colour_domain`, `image_pipeline`, `tactile_pair`) | yes |
| `norm_stats.json` | q01/q99 normalization statistics and the fixed 524/58 train/val split | yes |
| `fold_labels_auto.json` | automatic fold-stage boundaries (seconds) for all 582 training episodes | yes |
| `regime_labels.{json,csv}` | pre/post task-change tag for every downloaded season | yes |
| `ckpt/` | trained checkpoints (`best_<tag>.pt`); the final policy is `best_v10_sd50n10.pt` | no |
| `experiments/` | raw experiment outputs: Isaac Sim mirror trajectories and videos, replay tests, rollout calibration, Shadow-checker fixtures, training logs | no |
| `submission_bundle/`, `submission_upload/` | Docker build context and exported image archive (`scripts/build_submission_bundle.py`) | no |

Results are summarized in [`docs/RESULTS.md`](../docs/RESULTS.md) and on the
[project page](https://ismael-esp.github.io/robotic-origami-challenge/).
