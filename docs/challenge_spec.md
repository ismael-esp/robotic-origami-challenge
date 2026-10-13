# Robotic Origami Challenge — spec (IROS 2026)

Source: https://robotic-origami-challenge.github.io/

## Task
Fold a traditional Japanese paper airplane (*kami hikōki*) in **exactly 6 folds**.
Target figure is defined by the Nippon Origami Association; identical for all teams.

## Paper
- 15 × 15 cm, weight ≥ 60 gsm.

## Robot platform (identical in sim, remote eval, and on-site)
- Bimanual arms + **Sharpa Hands** dexterous end-effectors (teams don't bring hands).
- **6 synchronized video streams**: head_left, head_right, wrist_left, wrist_right,
  tactile_deform, tactile_raw.
- **65-D joint state/action** = left arm 7 + left hand 22 + right arm 7 +
  right hand 22 + torso 7.
- **10-fingertip 6-axis tactile** force/torque signals.

## Scoring
- **Pass/fail:** an Origami Grand Master (Nippon Origami Association) declares the
  fold recognizable or not.
- **Ranking among passes:** faster successful folds rank higher.
- **Time limit:** 10 minutes/attempt; exceeding = automatic fail.
- Whether the plane actually flies does NOT affect the score.
- Judged qualities: crease accuracy, structural fidelity, symmetry, paper integrity.

## Provided to registered teams
- **Dataset:** 500+ teleop demos (this repo: 682 episodes / 51 seasons, LeRobot v3.0).
- **Simulation:** NVIDIA **Isaac Sim**-based env with thin-shell paper physics
  (plastic creasing + fold memory), digital twins of the hands, same scoring rubric.
  *Released to registered teams* — not a public download.
- **Remote evaluation:** upload a policy, queue an eval, get scored on the real rig
  (no need to ship a robot to Pittsburgh).

## Logistics
- Event: IROS 2026, Sept 27 – Oct 1 2026, Pittsburgh PA (David L. Lawrence CC).
- First-edition registration is **closed**; next edition ~early 2026.
- Interest form: https://forms.gle/Y46WUArricVAQPBv9
- Contact: santiago.pravisani@frodobots.com
- Partners: BitRobot, Nippon Origami Association, Sharpa Hands, Lightwheel,
  Toyota Central R&D Labs.

## How to get the simulation
The Isaac Sim environment is **gated behind team registration**, same as the
dataset. There is no public URL. Action items:
1. Submit the interest form above.
2. Email the contact to confirm access to the sim package + figure spec + docs.
3. The dataset itself is already accessible once you accept its terms on HF
   (see `docs/getting_started.md`), but the simulator ships separately.
