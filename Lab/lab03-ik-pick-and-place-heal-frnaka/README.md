# ME639 Lab 3 — IK Pick-and-Place with HEAL and Franka Panda

**Course:** ME639 — Introduction to Robotics  
**Author:** Kshitij Suresh Giri  
**Task:** Implement and compare inverse-kinematics methods for cube pick-and-place in MuJoCo.

## Overview

This lab implements pick-and-place for the **HEAL** and **Franka Panda** arms and compares three IK methods:

- **Mink** — task-based IK baseline
- **DLS** — custom damped least-squares IK
- **QP** — custom quadratic-programming IK with joint constraints

The workflow includes workspace sampling, table selection, randomized episodes, experiment logging, and quantitative comparison.

## Submission Links

- **Demonstration video:** [Pick-and-Place Demo](https://youtu.be/yv1NYxJ2LtA)
- **Report:** `lab3_report_23110177.pdf`
- **Code and results:** Available in this directory.

## Repository Layout

```text
lab03-ik-pick-and-place-heal-frnaka/
├── README.md
├── scripts/
│   ├── heal_pick_place.py
│   ├── franka_pick_place.py
│   ├── heal_pick_place_scene.py
│   ├── franka_pick_place_scene.py
│   ├── compute_heal_task_workspace.py
│   ├── compute_heal_dexterous_workspace.py
│   ├── compute_franka_task_workspace.py
│   ├── compute_franka_dexterous_workspace.py
│   ├── run_experiments.py
│   └── supporting loader/control examples
├── utils/
│   ├── mink_ik/mink_ik.py
│   ├── dls_ik/dls_ik.py
│   ├── qp_ik/qp_ik.py
│   └── mj_velocity_control/mj_velocity_ctrl.py
├── robot-descriptions/
│   └── MuJoCo XML models, meshes, textures, and assets
├── results/
│   ├── workspace data, plots, and table-optimization notes
│   └── experiments/run_20261016_172020/
│       ├── episodes.csv / episodes.jsonl
│       ├── summary.csv / summary.json / report.md
│       ├── layouts.csv / failure_summary.csv / pairwise_tests.csv
│       ├── plots/ screenshots/ failure_states/ logs/
├── video-demo/
│   └── pick_place_video_demo.mp4
└── report/
    └── lab3_report_23110177.pdf

```

## Where to Start

| File or folder | Purpose |
|---|---|
| `scripts/heal_pick_place.py` | HEAL pick-and-place task runner. |
| `scripts/franka_pick_place.py` | Franka Panda pick-and-place task runner. |
| `scripts/*_pick_place_scene.py` | Builds the robot-specific MuJoCo scenes and task setup. |
| `utils/mink_ik/mink_ik.py` | Reusable Mink IK implementation. |
| `utils/dls_ik/dls_ik.py` | Reusable custom DLS IK implementation. |
| `utils/qp_ik/qp_ik.py` | Reusable custom QP IK implementation. |
| `utils/mj_velocity_control/mj_velocity_ctrl.py` | MuJoCo velocity-control helper. |
| `scripts/compute_*_workspace.py` | Workspace sampling and table-selection analysis. |
| `scripts/run_experiments.py` | Runs the seeded comparison across robots and IK methods. |
| `results/` | Workspace data/plots and saved experiment outputs. |

The IK implementations are separated into `utils/` so both robot-specific task scripts can reuse the same solver code. The `scripts/` folder contains task execution, scene setup, workspace analysis, and experiment orchestration.

## Workspace and Table Selection

Workspace scripts sample reachable end-effector targets and use the resulting feasible regions to guide table dimensions and height.

| Robot | Table dimensions (L × W) | Table-top height |
|---|---:|---:|
| HEAL | 0.45 × 0.82 m | 0.20 m |
| Franka Panda | 0.50 × 1.00 m | 0.10 m |

The analysis uses sampled reachability and task checks; it does not prove that every point on every continuous trajectory is collision-free. Randomized cube/tray layouts enforce a separation margin, and the motion lifts the cube before transferring it to the tray.

## Experiments and Results

The saved experiment run evaluates **150 episodes**: 2 robots × 3 IK methods × 25 seeds. For each robot, the same seeds are used across methods to support paired comparisons.

`results/experiments/run_20261016_172020/` contains:
- Per-episode records and randomized layout data
- Summary metrics and paired statistical tests
- Comparison plots
- Success/failure screenshots and failure-state images
- Individual run logs and the generated experiment summary report

Two Franka–DLS descent timeouts were recorded. Proposed mitigations are recommendations and were not verified by a rerun.

## Running the Code

From the lab directory, install the Python packages required by the scripts and ensure the robot-description assets are available. Example task commands:

```bash
python scripts/heal_pick_place.py
python scripts/franka_pick_place.py
```

Run the seeded experiment suite with:

```bash
python scripts/run_experiments.py
```

Workspace-analysis scripts are available as `scripts/compute_*_workspace.py`. Check each script's arguments before running a long experiment.

## Notes

- `robot-descriptions/` contains the MuJoCo XML models and assets required by the scenes.
- `results/` stores generated data and figures separately from source code.
- `video-demo/pick_place_video_demo.mp4` is the local video; the YouTube link above is the convenient viewing link.
