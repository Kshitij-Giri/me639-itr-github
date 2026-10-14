#!/usr/bin/env python3
"""HEAL pick-and-place with the custom DLS-IK solver (utils/dls_ik_heal/dls_ik_heal.py).

This is the *same task* as the Mink version. The whole task logic (scene, randomization incl. yaw,
velocity controller, gripper handling, stage sequence, tolerances, placement checks) is imported from
the Mink script (main_heal_mink.py, or main_heal.py if that is the file name in your repo); the ONLY
thing replaced is the IK object. The class below adds measurement hooks (no behaviour change).

Run from the repository root:
  python Lab/lab03-ik-pick-and-place-heal-frnaka/scripts/main_heal_dls.py [--seed N] [--no-viewer]
  python Lab/lab03-ik-pick-and-place-heal-frnaka/scripts/main_heal_dls.py --trials 20 --seed 0
  ... --ik mink   runs the unchanged Mink IK through the identical harness (baseline for comparison)
Other flags are forwarded to the task: --no-randomize, --no-random-yaw.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent
for _p in (LAB_DIR, SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import main_heal_mink as base
except ModuleNotFoundError as exc:  # repo uses the other file name
    if exc.name != "main_heal_mink":
        raise
    import main_heal as base

from utils.dls_ik_heal.dls_ik_heal import DLSIKHeal, IKFailure  # noqa: E402

IK_LABEL = "DLS (custom)"
MAX_CONSECUTIVE_IK_FAILURES = 50   # control ticks (0.5 s) of IK failure tolerated before the stage fails
_IK_KIND = "own"                   # "own" = custom solver, "mink" = unchanged Mink baseline


def make_ik(model, data, dt):
    """Custom solver with exactly the parameters the Mink task uses (task constants imported)."""
    return DLSIKHeal(model, data, site_name="ft_site_eff", dt=dt, damping=base.IK_DAMPING,
                     gain=base.IK_GAIN, max_joint_velocity=base.MAX_JOINT_SPEED)


class GuardedIK:
    """Same compute_joint_velocities(pos, rot) interface as MinkIKHeal; times every call.

    Failure policy (own solvers only): on IKFailure hold still for that tick and count it; after
    MAX_CONSECUTIVE_IK_FAILURES in a row raise TaskError so the episode fails loudly, never silently.
    """

    def __init__(self, ik, catch=()):
        self.ik, self.catch = ik, tuple(catch)
        self.calls = self.failures = self.consecutive = 0
        self.times = []

    def compute_joint_velocities(self, pos, rot):
        t0 = time.perf_counter()
        try:
            v = self.ik.compute_joint_velocities(pos, rot)
            self.consecutive = 0
            return v
        except self.catch as exc:
            self.failures += 1
            self.consecutive += 1
            if self.consecutive >= MAX_CONSECUTIVE_IK_FAILURES:
                raise base.TaskError(f"{IK_LABEL}: {self.consecutive} consecutive IK failures: {exc}") from exc
            return np.zeros(6)
        finally:
            self.calls += 1
            self.times.append(time.perf_counter() - t0)


class PickPlaceDLS(base.PickPlace):
    """Existing task, IK swapped, metrics recorded. No task behaviour is changed."""

    def __init__(self, model, data, viewer=None, realtime=True):
        super().__init__(model, data, viewer=viewer, realtime=realtime)  # builds the Mink IK too
        if _IK_KIND == "own":
            self.ik = GuardedIK(make_ik(model, data, self.dt), catch=(IKFailure,))
            self.ik_name = IK_LABEL
        else:
            self.ik = GuardedIK(self.ik)                                   # Mink, unchanged
            self.ik_name = "Mink (baseline)"
        self.metrics = {"ik": self.ik_name, "success": False}
        self.stages = []
        self._min_margin = np.inf
        self._lo = model.jnt_range[self.arm_joint_ids, 0]
        self._hi = model.jnt_range[self.arm_joint_ids, 1]

    def _advance(self, arm_velocity):
        super()._advance(arm_velocity)
        q = self.q_arm()
        self._min_margin = min(self._min_margin, float(np.min(np.minimum(q - self._lo, self._hi - q))))

    def move_tcp(self, label, goal, speed):
        t0 = self.data.time
        self.metrics["last_stage"] = label
        super().move_tcp(label, goal, speed)
        self.stages.append((label, self.data.time - t0,
                            1000.0 * float(np.linalg.norm(np.asarray(goal) - self.tcp())),
                            float(np.degrees(self.yaw_error())), float(np.degrees(self.tilt()))))

    def run(self):
        wall0, sim0 = time.perf_counter(), self.data.time
        try:
            super().run()
            self.metrics["success"] = True
        except Exception as exc:
            self.metrics["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._finalize(wall0, sim0)
            print_summary(self.metrics)

    def _finalize(self, wall0, sim0):
        m, cube, box = self.metrics, self.cube_pos(), self.box_pos()
        yaw_b = np.deg2rad(self.box_yaw_deg())
        dx, dy = cube[0] - box[0], cube[1] - box[1]
        lx, ly = np.cos(yaw_b) * dx + np.sin(yaw_b) * dy, -np.sin(yaw_b) * dx + np.cos(yaw_b) * dy
        t_ms = 1000.0 * np.asarray(self.ik.times or [0.0])
        m.update(
            sim_time_s=self.data.time - sim0, wall_time_s=time.perf_counter() - wall0,
            ik_calls=self.ik.calls, ik_failures=self.ik.failures, ik_mean_ms=float(t_ms.mean()),
            ik_p95_ms=float(np.percentile(t_ms, 95)), ik_max_ms=float(t_ms.max()),
            min_limit_margin_deg=float(np.degrees(self._min_margin)),
            max_stage_pos_err_mm=max([s[2] for s in self.stages], default=float("nan")),
            max_stage_yaw_err_deg=max([s[3] for s in self.stages], default=float("nan")),
            place_offset_mm=1000.0 * float(np.hypot(lx, ly)),
            place_yaw_err_deg=abs(base._wrap90(self.cube_yaw_deg() - self.box_yaw_deg())),
            return_joint_err_rad=float(np.max(np.abs(self.q_arm() - self.q_neutral))),
        )


def print_summary(m):
    print(f"\n----- METRICS [{m['ik']}] -----")
    print(f"success                 : {m['success']}" + (f"   ({m['error'][:120]})" if "error" in m else ""))
    for key, label, fmt in [
        ("sim_time_s", "task time (sim) [s]", "{:.2f}"), ("wall_time_s", "wall time [s]", "{:.2f}"),
        ("ik_calls", "IK calls", "{:d}"), ("ik_failures", "IK failures (held ticks)", "{:d}"),
        ("ik_mean_ms", "IK solve mean [ms]", "{:.3f}"), ("ik_p95_ms", "IK solve p95 [ms]", "{:.3f}"),
        ("min_limit_margin_deg", "min joint-limit margin [deg]", "{:.1f}"),
        ("max_stage_pos_err_mm", "worst stage TCP error [mm]", "{:.2f}"),
        ("max_stage_yaw_err_deg", "worst stage yaw error [deg]", "{:.2f}"),
        ("place_offset_mm", "cube offset from container centre [mm]", "{:.2f}"),
        ("place_yaw_err_deg", "cube yaw vs container walls [deg]", "{:.2f}"),
        ("return_joint_err_rad", "final joint error vs neutral [rad]", "{:.4f}"),
    ]:
        if key in m:
            print(f"{label:<39}: {fmt.format(m[key])}")


def run_trials(n, seed0, randomize, rand_yaw, verbose):
    rows = []
    for i in range(n):
        seed = seed0 + i
        quiet = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
        metrics = {"success": False, "error": "setup failed"}
        with quiet:
            try:
                model, data = base.build_world(randomize, seed, rand_yaw)
                runner = PickPlaceDLS(model, data, viewer=None, realtime=False)
                mujoco.set_mjcb_control(runner.controller.control_callback)
                try:
                    runner.run()
                except Exception:
                    pass  # recorded in runner.metrics
                finally:
                    mujoco.set_mjcb_control(None)
                metrics = runner.metrics
            except Exception as exc:
                metrics = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
        metrics["seed"] = seed
        rows.append(metrics)
        tag = "OK  " if metrics["success"] else "FAIL"
        extra = (f"sim {metrics['sim_time_s']:.1f}s, offset {metrics['place_offset_mm']:.2f} mm"
                 if "sim_time_s" in metrics else "")
        print(f"trial {i + 1:>3}/{n} seed {seed}: {tag} {extra}"
              + ("" if metrics["success"] else f"  [{metrics.get('error', '')[:100]}]"), flush=True)

    ok = [r for r in rows if r["success"]]

    def stat(key, rows_, fmt="{:.3f}"):
        vals = [r[key] for r in rows_ if key in r and np.isfinite(r[key])]
        return "n/a" if not vals else f"{fmt.format(np.mean(vals))} (max {fmt.format(np.max(vals))})"

    print(f"\n===== SUMMARY [{rows[0].get('ik', IK_LABEL)}] over {n} trials, seeds {seed0}..{seed0 + n - 1} =====")
    print(f"success rate               : {len(ok)}/{n} = {100.0 * len(ok) / n:.1f}%")
    print(f"task time, sim [s]         : {stat('sim_time_s', ok, '{:.2f}')}   (successful trials)")
    print(f"wall time [s]              : {stat('wall_time_s', rows, '{:.2f}')}")
    print(f"IK solve mean [ms]         : {stat('ik_mean_ms', rows)}   p95 {stat('ik_p95_ms', rows)}")
    print(f"IK failures (held ticks)   : {sum(r.get('ik_failures', 0) for r in rows)}")
    print(f"placement offset [mm]      : {stat('place_offset_mm', ok, '{:.2f}')}")
    print(f"placement yaw error [deg]  : {stat('place_yaw_err_deg', ok, '{:.3f}')}")
    print(f"worst stage TCP err [mm]   : {stat('max_stage_pos_err_mm', rows, '{:.2f}')}")
    print(f"return joint error [rad]   : {stat('return_joint_err_rad', ok, '{:.4f}')}")
    print(f"min joint-limit margin [deg]: {min([r['min_limit_margin_deg'] for r in rows if 'min_limit_margin_deg' in r], default=float('nan')):.1f}")
    return 0 if len(ok) == n else 1


def main():
    global _IK_KIND
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ik", choices=["own", "mink"], default="own", help="own = this script's solver; mink = baseline")
    ap.add_argument("--trials", type=int, default=1, help="N>1: headless benchmark over seeds seed..seed+N-1")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--no-randomize", action="store_true")
    ap.add_argument("--no-random-yaw", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="benchmark mode: print every stage log")
    args = ap.parse_args()
    _IK_KIND = args.ik

    if args.trials > 1:
        sys.exit(run_trials(args.trials, 0 if args.seed is None else args.seed,
                            not args.no_randomize, not args.no_random_yaw, args.verbose))

    # Single episode: run the Mink task's own main() (viewer, CLI, exit codes) with our class swapped in.
    forwarded = [sys.argv[0]]
    for flag in ("no_viewer", "no_randomize", "no_random_yaw"):
        if getattr(args, flag):
            forwarded.append("--" + flag.replace("_", "-"))
    if args.seed is not None:
        forwarded += ["--seed", str(args.seed)]
    sys.argv = forwarded
    base.PickPlace = PickPlaceDLS
    base.main()


if __name__ == "__main__":
    main()
