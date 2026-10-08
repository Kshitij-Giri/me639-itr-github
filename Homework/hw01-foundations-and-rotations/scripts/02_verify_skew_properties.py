"""
02_verify_skew_properties.py -- HW1 Part 2, Task 2: verify the
skew-symmetric identities from Problem 5 in simulation.

STARTER CODE. Model loading and a simple "spin the body" simulation
loop are provided and working. Your job is to fill in the TODOs to:

  1. Log R(t), the body's rotation matrix, at several simulated
     time steps while it spins.
  2. At each logged time step, numerically check, for several
     random v, w, omega in R^3:
         R (v x w) == (R v) x (R w)                 [Problem 5a]
         R w^ R^T  == (R w)^                         [Problem 5b, No-AI on paper]
     using utils.hat() for the ^ operator.
  3. Print the residual (it should be ~1e-14, machine precision)
     and explain in your write-up why a small-but-nonzero residual
     doesn't fully validate the identity, while a residual near
     machine epsilon strongly supports it.

Note: you already proved these identities by hand in Problem 5.
This script is not a substitute for that proof -- it's a numerical
sanity check, and a chance to see *why* proofs and simulation are
complementary, not interchangeable.
"""

import os
import sys
import warnings

warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning
)

# Above section is added to suppress warnings that are not relevant to the assignment.

import numpy as np
import mujoco
import matplotlib.pyplot as plt

from pathlib import Path

from utils import hat, get_body_orientation, is_close_to_identity

# Resolve paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
HW01_DIR = SCRIPT_DIR.parent

MODEL_PATH = HW01_DIR / "model" / "asymmetric_body.xml"
OUTPUT_DIR = SCRIPT_DIR

N_CHECKS_PER_STEP = 5     # how many random (v, w, omega) triples per logged step
N_LOGGED_STEPS = 5        # how many simulated time points to check
STEPS_BETWEEN_LOGS = 200  # sim steps to advance between each logged check


def random_unit_angular_velocity(rng):
    """A random constant angular velocity vector (rad/s), used to spin
    the body between logged checks."""
    w = rng.normal(size=3)
    return 2.0 * w / np.linalg.norm(w)

def time_varying_angular_velocity(t):
    """Return a smooth time-varying angular velocity vector."""
    return np.array([
        2.0 * np.sin(t),
        2.0 * np.cos(t),
        1.0 + 0.5 * np.sin(2.0 * t)
    ])

def check_identities(R, rng):
    # TODO(student): implement this.

    # For N_CHECKS_PER_STEP random vectors v, w, omega (use rng.normal
    # or rng.uniform), compute the residuals of:
    #     R @ np.cross(v, w)  vs.  np.cross(R @ v, R @ w)
    #     R @ hat(omega) @ R.T  vs.  hat(R @ omega)
    # and return the worst-case (max) residual across all checks, for
    # each identity separately.

    # Return: (max_residual_cross, max_residual_skew)

    # --> Implemented the numerical checks below.
    
    max_residual_cross = 0.0
    max_residual_skew = 0.0

    for _ in range(N_CHECKS_PER_STEP):

        # Generate random vectors
        v = rng.normal(size=3)
        w = rng.normal(size=3)
        omega = rng.normal(size=3)

        # -------------------------------------------------------
        # Identity 1:
        #
        # R(v x w) = (Rv) x (Rw)
        # -------------------------------------------------------

        lhs_cross = R @ np.cross(v, w)
        rhs_cross = np.cross(R @ v, R @ w)

        residual_cross = np.linalg.norm(lhs_cross - rhs_cross)

        # Keep the worst residual
        max_residual_cross = max(max_residual_cross,residual_cross)

        # -------------------------------------------------------
        # Identity 2:
        #
        # R hat(omega) R^T = hat(R omega)
        # -------------------------------------------------------

        lhs_skew = R @ hat(omega) @ R.T
        rhs_skew = hat(R @ omega)

        residual_skew = np.linalg.norm(lhs_skew - rhs_skew)

        # Keep the worst residual
        max_residual_skew = max(max_residual_skew, residual_skew)

    return max_residual_cross, max_residual_skew

def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    
    rng = np.random.default_rng(seed=0)

    choice = input("Angular velocity mode ([T] Time-varying / [F] Fixed): ").strip().lower()
    if choice == "t":
        # Spin the body with a time-varying angular velocity by directly setting
        # qvel's angular part (indices 3:6 for a freejoint) and stepping.
        data.qvel[3:6] = time_varying_angular_velocity(data.time)
        print("Spinning the body with a time-varying angular velocity.")
    elif choice == "f":
        # Spin the body with a fixed angular velocity by directly setting
        # qvel's angular part (indices 3:6 for a freejoint) and stepping.
        data.qvel[3:6] = random_unit_angular_velocity(rng)
        print("Spinning the body with a fixed angular velocity.")
    else:
        print("Invalid choice. Please enter 'T' or 'F'.")
        return

    mujoco.mj_forward(model, data)

    print(f"{'step':>5} {'t (s)':>8} {'max resid: R(vxw)=(Rv)x(Rw)':>28} {'max resid: RwR^T=(Rw)^':>24}")

    results = []

    for log_i in range(N_LOGGED_STEPS):
        for _ in range(STEPS_BETWEEN_LOGS):
            if choice == "t":
                data.qvel[3:6] = time_varying_angular_velocity(data.time)
            mujoco.mj_step(model, data)

        R = get_body_orientation(data)
        # Sanity check that R is actually a valid rotation matrix
        # (this should hold to numerical precision -- if it doesn't,
        # something upstream is wrong before you even get to the
        # identities below).
        assert is_close_to_identity(R @ R.T, tol=1e-6), "R is not orthonormal!"

        resid_cross, resid_skew = check_identities(R, rng)
        print(f"{log_i:5d} {data.time:8.3f} {resid_cross:28.3e} {resid_skew:24.3e}")
        results.append([log_i, data.time, resid_cross, resid_skew])

    mode_name = "time_varying" if choice == "t" else "fixed"

    # Saved the residuals to a CSV file for further analysis and plotting.
    results = np.array(results)
    np.savetxt(OUTPUT_DIR / f"residuals_{mode_name}.csv", results, delimiter=",", header="step,time,resid_cross,resid_skew", comments="")
    print(f"\nResiduals saved to residuals_{mode_name}.csv.")

    # Plot the residuals for visual inspection, saved the figure plot in PNG format.
    times = results[:, 1]
    cross_residuals = results[:, 2]
    skew_residuals = results[:, 3]

    plt.figure()

    plt.semilogy(
        times,
        cross_residuals,
        marker="o",
        label=r"$R(v\times w) - (Rv)\times(Rw)$"
    )

    plt.semilogy(
        times,
        skew_residuals,
        marker="s",
        label=r"$R\hat{\omega}R^T - \widehat{R\omega}$"
    )

    plt.xlabel("Simulation time (s)")
    plt.ylabel("Maximum residual")
    plt.title("Numerical Verification of Rotation Identities")
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"residuals_{mode_name}_plot.png", dpi=300, bbox_inches="tight")
    print(f"Residuals plotted and saved to residuals_{mode_name}_plot.png.")
    plt.show()

    # TODO(student): save these residuals (e.g. to a CSV or a plot)
    # for your write-up, and answer the "why doesn't a small nonzero
    # residual fully prove the identity" question from HW1 Problem 8.

    # --> Implemented above.
    # For writeup see explanation below.

if __name__ == "__main__":
    main()


# Explanation:
# The residuals are very small (around 1e-15), which is close to
# floating-point machine precision. This strongly supports that the
# two identities are being satisfied numerically.
#
# However, a small nonzero residual does not mathematically prove
# the identities. The code only tests a finite number of randomly 
# selected vectors and a finite number of rotation matrices at
# particular simulation times. The identities are universal statements 
# which should be true for the possible vectors and rotation matrices.
#
# A small but nonzero residual usually means the identity holds to 
# numerical precision, with the small error caused by floating-point 
# rounding. It provides strong numerical evidence for the identity, 
# but does not mathematically prove it.