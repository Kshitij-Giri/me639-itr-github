# Pick-and-place IK comparison: run_20261016_172020

Generated 20261016_174952. Episodes recorded: 150. Seeds per robot: 25 (identical across IK methods; same for both robots).

Success means the task script's own `run()` finished without raising: placement verified inside the container (position, resting height and wall-parallel yaw) and return to neutral completed.

IK (planning) time is the wall-clock time spent inside the solver's `compute_joint_velocities` only; episode time is the full wall-clock execution of `run()` (simulation, control and IK).


## Summary

| Robot | IK | n | Success | 95% CI | IK time/episode (s) | IK/call (ms) | Episode time (s) | Min clearance (mm) | Pos-limit viol. eps | Speed viol. eps |
|---|---|---|---|---|---|---|---|---|---|---|
| HEAL | Mink | 25 | 25/25 (100%) | 87-100% | 1.822 | 1.080 | 18.8 | 3.6 | 0 | 7 |
| HEAL | DLS | 25 | 25/25 (100%) | 87-100% | 0.539 | 0.321 | 16.9 | 3.6 | 0 | 7 |
| HEAL | QP | 25 | 25/25 (100%) | 87-100% | 0.720 | 0.429 | 16.3 | 3.6 | 0 | 7 |
| Franka Panda | Mink | 25 | 25/25 (100%) | 87-100% | 0.765 | 0.410 | 5.4 | 14.3 | 0 | 0 |
| Franka Panda | DLS | 25 | 23/25 (92%) | 75-98% | 0.283 | 0.152 | 4.8 | 14.3 | 2 | 0 |
| Franka Panda | QP | 25 | 25/25 (100%) | 87-100% | 0.401 | 0.215 | 5.0 | 14.3 | 0 | 0 |

IK iterations: unavailable - all three solvers are differential IK methods that perform exactly one solve per control step and do not expose internal solver iterations; `ik_solver_calls` is reported instead.


## Failure analysis

| Robot | IK | Category | Count | Stages | Example seeds | Mitigation |
|---|---|---|---|---|---|---|
| Franka Panda | DLS | motion_timeout_descend | 2 | ["STAGE 2"] | 316063302 1471704793 | Target unreachable or tracking too slow with the required yaw/tilt: check workspace reach at this layout, joint limits (QP active limits), singularity diagnostics, and consider relaxing POS_TOL or reducing speed. |

## Paired comparison (same seeds, exact McNemar test)

| Robot | A | B | Seeds | A only | B only | Both | Neither | p |
|---|---|---|---|---|---|---|---|---|
| HEAL | Mink | DLS | 25 | 0 | 0 | 25 | 0 | 1.000 |
| HEAL | Mink | QP | 25 | 0 | 0 | 25 | 0 | 1.000 |
| HEAL | DLS | QP | 25 | 0 | 0 | 25 | 0 | 1.000 |
| Franka Panda | Mink | DLS | 25 | 2 | 0 | 23 | 0 | 0.500 |
| Franka Panda | Mink | QP | 25 | 0 | 0 | 25 | 0 | 1.000 |
| Franka Panda | DLS | QP | 25 | 0 | 2 | 23 | 0 | 0.500 |

Layout consistency: 0 episode(s) had an initial layout different from the reference for their seed.

## Workspace / table

- HEAL: tabletop X 0.120..0.570 m, Y -0.410..0.410 m, top z = 0.200 m (see plots/heal_layouts_workspace.png).
- Franka Panda: tabletop X 0.200..0.700 m, Y -0.500..0.500 m, top z = 0.100 m (see plots/franka_layouts_workspace.png).

## Files

- episodes.csv / episodes.jsonl, summary.csv, failure_summary.csv, pairwise_tests.csv, layouts.csv, config.json
- plots/, failure_states/, screenshots/, logs/
