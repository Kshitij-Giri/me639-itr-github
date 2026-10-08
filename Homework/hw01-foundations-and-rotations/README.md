# ME 639 -- Introduction to Robotics

## Homework 1: Foundations & Rotations

**AY 2026--27 \| IIT Gandhinagar**\
**Instructor:** Madhu Vadali\
**Homework:** HW1 -- Foundations & Rotations

------------------------------------------------------------------------

## Overview

This directory contains my complete submission for **ME 639 --
Introduction to Robotics, Homework 1: Foundations & Rotations**.

The homework has two parts:

-   **Part 1 -- Pen and Paper:** Problems 1--6, covering vector
    identities, coordinate frames, rotation matrices, composition of
    rotations, properties of `SO(3)` and `so(3)`, and `SO(2)`.
-   **Part 2 -- Implementation & Visualization:** Problems 7--9,
    implementing and visualizing rotation concepts using **MuJoCo** and
    **ROS 2 / RViz2**.

The implementation is organized so that the repository does not depend
on PC-specific absolute paths.

The official assignment is included as `hw01.pdf`.

------------------------------------------------------------------------

# Assignment Requirements

## Part 1 -- Pen and Paper

Problems 1--6 cover:

1.  Vector and cross-product identities
2.  Frames, rotation matrices, and what they represent
3.  Composition of three basic rotations
4.  Proof of `R_ab R_bc = R_ac`
5.  Properties of rotation matrices and `so(3)`
6.  The planar rotation group `SO(2)`

The assignment emphasizes showing the reasoning/work rather than only
final answers.

## Part 2 -- Implementation & Visualization

Problems 7--9 cover:

7.  **Rotation sandbox:** current-frame vs. fixed-frame rotation
    composition in MuJoCo
8.  **Skew-symmetric identities:** numerical verification using a
    simulated time-varying rotation
9.  **ROS/RViz:** live visualization of current-frame vs. fixed-frame
    composition

------------------------------------------------------------------------

# Repository Structure

``` text
hw01-foundations-and-rotations/
│
├── ai-usenote/
│   ├── Q7.txt
│   ├── Q8.txt
│   └── Q9.txt
│
├── model/
│   ├── asymmetric_body.xml
│   └── Q1-Q6/
│       └── Q1-Q6.pdf
│
├── ros_ws/
│   ├── build/
│   ├── install/
│   ├── log/
│   ├── src/
│   └── README.md
│
├── scripts/
│   ├── 01_rotation_sandbox.py
│   ├── 02_verify_skew_properties.py
│   ├── utils.py
│   ├── residuals_fixed.csv
│   ├── residuals_fixed_plot.png
│   ├── residuals_time_varying.csv
│   └── residuals_time_varying_plot.png
│
├── video-submissions/
│   ├── Q7.mp4
│   └── Q9.mp4
│
├── hw01.pdf
├── README.md
└── requirements.txt
```

------------------------------------------------------------------------

# Submission Contents

## Problems 1--6

### `model/Q1-Q6/Q1-Q6.pdf`

This PDF contains the submitted solutions for **Problems 1--6**.

It covers the mathematical foundations of the homework, including vector
identities, frame representations, rotation matrices, inverse rotations,
current-frame vs. fixed-frame rotation, point and angular-velocity
transformations, composition of rotations, `SO(3)`, `so(3)`, and
`SO(2)`.

------------------------------------------------------------------------

# Problem 7 --- Rotation Sandbox

## Files

``` text
scripts/
├── 01_rotation_sandbox.py
└── utils.py

model/
└── asymmetric_body.xml

video-submissions/
└── Q7.mp4
```

## Objective

Problem 7 asks for an asymmetric MuJoCo body to undergo a sequence of
elemental rotations about the `x`, `y`, and `z` axes, with each rotation
applied either about the current/body frame or the fixed/space frame.

The two composition conventions are:

**Current/body frame**

``` text
R_new = R_current @ R_step
```

**Fixed/space frame**

``` text
R_new = R_step @ R_current
```

Since 3D rotations generally do not commute, the two conventions can
produce different final orientations.

The asymmetric body makes the orientation difference visually clear.

## Run

From this HW01 directory:

``` bash
cd scripts
python3 01_rotation_sandbox.py
```

The script loads the model using a repository-relative path and opens
the MuJoCo viewer.

The submitted demonstration is:

``` text
video-submissions/Q7.mp4
```

------------------------------------------------------------------------

# Problem 8 --- Verifying the Skew-Symmetric Identities

## Files

``` text
scripts/
├── 02_verify_skew_properties.py
├── utils.py
├── residuals_fixed.csv
├── residuals_fixed_plot.png
├── residuals_time_varying.csv
└── residuals_time_varying_plot.png
```

## Objective

Problem 8 asks the simulation to generate a time-varying rotation matrix
`R(t)` and numerically verify:

``` text
R(v × w) = (Rv) × (Rw)
```

and

``` text
R hat(ω) Rᵀ = hat(Rω)
```

The identities are evaluated at multiple simulated time steps using
several vectors.

## Numerical results

The numerical residuals are stored in:

``` text
residuals_fixed.csv
residuals_time_varying.csv
```

The corresponding plots are:

``` text
residuals_fixed_plot.png
residuals_time_varying_plot.png
```

The residuals are very close to machine precision, as expected for
numerical floating-point evaluation.

A small residual does not constitute a universal mathematical proof. It
shows that the identities hold for the sampled cases, up to numerical
precision.

## Run

``` bash
cd scripts
python3 02_verify_skew_properties.py
```

------------------------------------------------------------------------

# Problem 9 --- Current Frame vs. Fixed Frame in ROS/RViz

## Files

``` text
ros_ws/
├── src/
│   └── hw01_tf_demo/
│       └── ...
└── README.md

video-submissions/
└── Q9.mp4
```

## Objective

Problem 9 implements the same current-frame vs. fixed-frame rotation
concept using **ROS 2 TF** and **RViz2**.

The implementation:

-   broadcasts the space frame
-   broadcasts the body frame
-   applies the rotation sequence used in the earlier task
-   updates the body orientation over time
-   supports switching between current-frame and fixed-frame composition
-   visualizes the result live in RViz2

The composition mode is controlled through the ROS parameter:

``` text
compose_frame
```

with:

``` text
current
fixed
```

------------------------------------------------------------------------

# ROS 2 Setup

The ROS implementation was tested with **ROS 2 Humble**.

First source ROS:

``` bash
source /opt/ros/humble/setup.bash
```

Enter the workspace:

``` bash
cd ros_ws
```

Build:

``` bash
colcon build
```

Then source the generated workspace:

``` bash
source install/setup.bash
```

------------------------------------------------------------------------

# Running Problem 9

## Terminal 1 --- TF Broadcaster

``` bash
source /opt/ros/humble/setup.bash

cd ros_ws
source install/setup.bash

ros2 run hw01_tf_demo tf_broadcaster_node
```

Leave this terminal running.

## Terminal 2 --- RViz2

Open another terminal:

``` bash
source /opt/ros/humble/setup.bash

source ~/Desktop/Work/me639-itr-github/Homework/hw01-foundations-and-rotations/ros_ws/install/setup.bash

rviz2
```

In RViz2:

1.  Set **Global Options → Fixed Frame** to `space_frame`.
2.  Click **Add**.
3.  Add the **TF** display.

The published frames can then be observed live.

------------------------------------------------------------------------

# Switching Composition Mode

While the node is running:

### Current frame

``` bash
ros2 param set /hw01_tf_broadcaster compose_frame current
```

This uses:

``` text
R_new = R_current @ R_step
```

### Fixed frame

``` bash
ros2 param set /hw01_tf_broadcaster compose_frame fixed
```

This uses:

``` text
R_new = R_step @ R_current
```

The difference can therefore be observed live in RViz2.

The node also includes keyboard controls for changing the mode:

``` text
c → current
f → fixed
```

------------------------------------------------------------------------

# TF Frames

The implementation publishes the relevant coordinate frames for
visualization.

Conceptually:

``` text
world
├── space_frame
└── body_frame
```

The space frame remains fixed while the body frame changes orientation
according to the selected composition convention.

------------------------------------------------------------------------

# Utility Functions

Common rotation functionality is kept in:

``` text
scripts/utils.py
```

The utilities include operations for:

-   `hat`
-   `vee`
-   `Rx`, `Ry`, `Rz`
-   quaternion/rotation-matrix conversions
-   reading and setting body orientation
-   other rotation-related helper functions

The task scripts reuse these functions instead of duplicating the
rotation operations.

------------------------------------------------------------------------

# MuJoCo Model

The MuJoCo model is:

``` text
model/asymmetric_body.xml
```

The body is intentionally asymmetric so that its orientation can be
identified visually.

------------------------------------------------------------------------

# Reproducibility

The Python scripts use paths relative to the repository structure rather
than machine-specific absolute paths.

For example, the model is located relative to:

``` text
hw01-foundations-and-rotations/
├── model/
└── scripts/
```

Therefore the HW01 directory can be moved or cloned to another location
without changing the model path in the Python scripts.

The ROS implementation is similarly contained inside:

``` text
ros_ws/
```

and follows the standard ROS 2 `colcon` build/source workflow.

------------------------------------------------------------------------

# Dependencies

Python dependencies are listed in:

``` text
requirements.txt
```

A fresh Python environment can be created using:

``` bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then run the MuJoCo scripts from `scripts/`.

------------------------------------------------------------------------

# Quick Start

## Problem 7

``` bash
cd Homework/hw01-foundations-and-rotations/scripts
python3 01_rotation_sandbox.py
```

## Problem 8

``` bash
cd Homework/hw01-foundations-and-rotations/scripts
python3 02_verify_skew_properties.py
```

## Problem 9

### Terminal 1

``` bash
source /opt/ros/humble/setup.bash
cd Homework/hw01-foundations-and-rotations/ros_ws
colcon build
source install/setup.bash
ros2 run hw01_tf_demo tf_broadcaster_node
```

### Terminal 2

``` bash
source /opt/ros/humble/setup.bash
source ~/Desktop/Work/me639-itr-github/Homework/hw01-foundations-and-rotations/ros_ws/install/setup.bash
rviz2
```

Then set the RViz Fixed Frame to:

``` text
space_frame
```

and add the **TF** display.

------------------------------------------------------------------------

# Submission Map

  -------------------------------------------------------------------------------
  Assignment Component                Location
  ----------------------------------- -------------------------------------------
  Problems 1--6 solutions             `model/Q1-Q6/Q1-Q6.pdf`

  Problem 7 implementation            `scripts/01_rotation_sandbox.py`

  Problem 7 model                     `model/asymmetric_body.xml`

  Problem 7 video                     `video-submissions/Q7.mp4`

  Problem 8 implementation            `scripts/02_verify_skew_properties.py`

  Problem 8 utilities                 `scripts/utils.py`

  Problem 8 fixed-mode residuals      `scripts/residuals_fixed.csv`

  Problem 8 fixed-mode plot           `scripts/residuals_fixed_plot.png`

  Problem 8 time-varying residuals    `scripts/residuals_time_varying.csv`

  Problem 8 time-varying plot         `scripts/residuals_time_varying_plot.png`

  Problem 9 ROS workspace             `ros_ws/`

  Problem 9 ROS instructions          `ros_ws/README.md`

  Problem 9 video                     `video-submissions/Q9.mp4`

  Problem 7 AI Use Note               `ai-usenote/Q7.txt`

  Problem 8 AI Use Note               `ai-usenote/Q8.txt`

  Problem 9 AI Use Note               `ai-usenote/Q9.txt`

  Original assignment                 `hw01.pdf`
  -------------------------------------------------------------------------------

------------------------------------------------------------------------

# Verification Status

The implementation has been tested locally.

### MuJoCo

-   [x] Problem 7 script runs
-   [x] Asymmetric-body model loads
-   [x] Current-frame composition works
-   [x] Fixed-frame composition works
-   [x] Rotation behaviour is visually distinguishable
-   [x] Problem 7 video included

### Numerical Verification

-   [x] Problem 8 script runs
-   [x] Rotation matrices are sampled during simulation
-   [x] Cross-product identity is numerically checked
-   [x] Skew-symmetric identity is numerically checked
-   [x] Residual CSV files included
-   [x] Residual plots included

### ROS 2 / RViz2

-   [x] ROS 2 Humble environment verified
-   [x] `ros_ws` builds with `colcon`
-   [x] `hw01_tf_demo` package builds and is discoverable
-   [x] `tf_broadcaster_node` runs
-   [x] TF frames are visualized in RViz2
-   [x] Current/fixed composition can be switched live
-   [x] Problem 9 video included

------------------------------------------------------------------------

# AI Use Notes

The official homework requires AI use to be disclosed when AI was used.

The submitted notes are stored in:

``` text
ai-usenote/
├── Q7.txt
├── Q8.txt
└── Q9.txt
```

These files document the AI assistance associated with the corresponding
implementation tasks.

The assignment's Part 2 policy explicitly allows AI use, including vibe
coding, while requiring disclosure and reflection.

------------------------------------------------------------------------

# Notes

-   `build/`, `install/`, and `log/` are generated ROS workspace
    directories.
-   The source of the ROS implementation is under `ros_ws/src/`.
-   The MuJoCo scripts can be run from the `scripts/` directory using
    the commands above.
-   The original assignment is retained as `hw01.pdf` for reference.
-   The repository is intended to be understandable and reproducible
    without relying on the original machine's absolute directory paths.

------------------------------------------------------------------------

## Author

**Kshitij Suresh Giri**\
B.Tech Dual Major --- Mechanical Engineering & Artificial Intelligence\
Indian Institute of Technology Gandhinagar

------------------------------------------------------------------------

## Course

**ME 639 -- Introduction to Robotics**\
IIT Gandhinagar\
AY 2026--27