# ME 639 -- Introduction to Robotics

## IIT Gandhinagar

This repository is a **course-wide collection of homework,
laboratory work, demonstrations, robot models, and supporting
resources** for **ME 639 -- Introduction to Robotics** at IIT
Gandhinagar.

The repository is organized so that each major course component is kept
in its own directory. Each homework and lab folder is intended to be
**self-contained**, with its own README explaining what is included, how
the work is organized, and how to run the relevant code.

The goal is simple:

> **Open the repository → choose the relevant Homework/Lab/Resource →
> read its README → follow the instructions.**

You should not need to understand the entire repository before working
on a particular assignment.

------------------------------------------------------------------------

# Repository Navigation

The main repository is organized as follows:

``` text
me639-itr-github/
│
├── Homework/
│   ├── hw00-interactive-3d-frame/
│   └── hw01-foundations-and-rotations/
│
├── Lab/
│   ├── lab01-mujoco-intro/
│   ├── lab02-fk-heal-franka/
│   └── lab03-ik-pick-and-place-heal-franka/
│
├── Robot-Descriptions/
│   ├── mujoco_menagerie/
│   └── robotis_mujoco_menagerie/
│
├── Textbook/
│   └── Spong-RobotmodelingandControl.pdf
│
└── README.md
```

The repository may grow as the course progresses. New homework, labs,
notes, demonstrations, or resources can be added without changing the
overall navigation philosophy.

------------------------------------------------------------------------

# Where Should I Start?

## If you are looking for a Homework

Go to:

``` text
Homework/
```

Then open the homework you want.

For example:

``` text
Homework/
└── hw01-foundations-and-rotations/
```

Each homework directory contains its own README with:

-   assignment overview
-   folder structure
-   problem/task mapping
-   required dependencies
-   execution instructions
-   submission files
-   demonstrations/videos where applicable
-   AI-use information where applicable

**Always read the README inside the specific homework folder first.**

------------------------------------------------------------------------

## If you are looking for a Lab

Go to:

``` text
Lab/
```

The current lab directories are:

``` text
Lab/
├── lab01-mujoco-intro/
├── lab02-fk-heal-franka/
└── lab03-ik-pick-and-place-heal-franka/
```

Each lab is kept as a separate self-contained unit.

The lab README explains:

-   what the lab does
-   the tasks performed
-   where the source code is
-   where robot descriptions are located
-   how to run the code
-   demonstrations/videos
-   any required setup

------------------------------------------------------------------------

## If you are looking for Robot Models

Go to:

``` text
Robot-Descriptions/
```

This directory contains reusable robot descriptions and related MuJoCo
resources.

Current collections include:

``` text
Robot-Descriptions/
├── mujoco_menagerie/
└── robotis_mujoco_menagerie/
```

These resources are kept separately from individual homework/lab code so
that robot descriptions can be reused across multiple course activities.

Individual labs may also contain local robot-description folders when a
specific version/model is required for that lab.

------------------------------------------------------------------------

## If you are looking for the Textbook

Go to:

``` text
Textbook/
```

The repository currently includes:

``` text
Textbook/
└── Spong-RobotmodelingandControl.pdf
```

This is provided as a course reference resource.

------------------------------------------------------------------------

# Homework

## HW00 --- Interactive 3D Frame Rotation Visualizer

Location:

``` text
Homework/hw00-interactive-3d-frame/
```

HW00 contains an interactive web-based 3D frame/rotation visualization.

The folder contains the web implementation, README, and demonstration
material.

The README inside the folder provides the details for using the
interactive visualization.

------------------------------------------------------------------------

# HW01 --- Foundations & Rotations

Location:

``` text
Homework/hw01-foundations-and-rotations/
```

HW01 covers the mathematical and computational foundations of rotations.

It contains two broad components:

### Part 1 --- Foundations

The pen-and-paper portion covers:

1.  Vector and cross-product identities
2.  Frames and rotation matrices
3.  Composition of basic rotations
4.  The property

``` text
R_ab R_bc = R_ac
```

5.  Properties of rotation matrices and `so(3)`
6.  The planar rotation group `SO(2)`

### Part 2 --- Implementation & Visualization

The implementation portion covers:

-   MuJoCo rotation composition
-   current-frame vs. fixed-frame rotations
-   numerical verification of rotation identities
-   ROS 2 TF
-   RViz2 visualization

The HW01 directory contains:

``` text
Homework/hw01-foundations-and-rotations/
├── ai-usenote/
├── model/
├── ros_ws/
├── scripts/
├── video-submissions/
├── hw01.pdf
├── README.md
└── requirements.txt
```

The **HW01-specific README** contains the detailed execution
instructions.

For HW01, start here:

``` text
Homework/hw01-foundations-and-rotations/README.md
```

------------------------------------------------------------------------

# Labs

## Lab 01 --- MuJoCo Introduction

Location:

``` text
Lab/lab01-mujoco-intro/
```

This lab introduces simulation in MuJoCo and contains demonstrations
involving:

-   a quadrotor model
-   a TurtleBot3 Waffle model

The lab contains its own:

-   source code
-   robot descriptions
-   explanation material
-   video demonstrations
-   README

Start with:

``` text
Lab/lab01-mujoco-intro/README.md
```

The code uses repository-relative paths so that the models can be loaded
without relying on the original computer's absolute directory structure.

------------------------------------------------------------------------

## Lab 02 --- Forward Kinematics: HEAL & Franka

Location:

``` text
Lab/lab02-fk-heal-franka/
```

This lab focuses on forward kinematics for:

-   Franka
-   HEAL

It contains work involving:

-   DH-based forward kinematics
-   MuJoCo-based forward kinematics
-   comparison of the implementations
-   robot descriptions
-   task solutions
-   demonstration video

The lab contains separate task directories and a dedicated README.

Start with:

``` text
Lab/lab02-fk-heal-franka/README.md
```

------------------------------------------------------------------------

## Lab 03 --- IK Pick-and-Place: HEAL & Franka

Location:

``` text
Lab/lab03-ik-pick-and-place-heal-franka/
```

This lab contains the inverse-kinematics / pick-and-place work for the
HEAL and Franka robots.

The lab is kept as its own self-contained directory.

Start with:

``` text
Lab/lab03-ik-pick-and-place-heal-franka/README.md
```

------------------------------------------------------------------------

# General Philosophy of the Repository

The repository follows a few simple principles.

## 1. Each assignment is self-contained

A user should be able to enter a homework or lab directory and
understand what is required without reading the entire repository.

For example:

``` text
Homework/hw01-foundations-and-rotations/
```

has its own README.

------------------------------------------------------------------------

## 2. Shared resources are separated from assignment work

Reusable robot descriptions are kept under:

``` text
Robot-Descriptions/
```

while assignment-specific models can remain inside their corresponding
homework/lab directory.

This avoids unnecessarily duplicating shared resources.

------------------------------------------------------------------------

## 3. Code should avoid PC-specific paths

Where possible, code uses paths relative to the repository or the script
location rather than hard-coded paths such as:

``` text
/home/username/Desktop/...
```

This makes the repository easier to:

-   clone
-   move
-   fork
-   share
-   reproduce on another machine

If you encounter a script that requires a machine-specific path, check
that assignment's README before modifying anything.

------------------------------------------------------------------------

## 4. Every major folder should explain itself

The intended navigation pattern is:

``` text
Main README
     ↓
Homework / Lab / Resource
     ↓
Folder README
     ↓
Code / Models / Results / Videos
```

The main README is intentionally general so it does not need to be
rewritten every time a new file is added.

------------------------------------------------------------------------

# Running Code

There is no single command that runs the entire repository.

Different course activities use different tools and environments.

Always check the README for the specific homework or lab before running
code.

Common environments used in the repository include:

-   Python
-   MuJoCo
-   ROS 2
-   RViz2
-   web-based HTML/JavaScript demonstrations

------------------------------------------------------------------------

# Python Projects

For Python-based assignments, the recommended pattern is to create an
environment inside the relevant assignment directory when required.

Typical setup:

``` bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The exact dependencies and commands may differ between assignments, so
the local README should take precedence.

------------------------------------------------------------------------

# MuJoCo Projects

For MuJoCo-based work:

1.  Open the relevant lab/homework directory.
2.  Read its README.
3.  Install the required Python dependencies.
4.  Run the provided script from the location specified by the README.

For example, HW01 contains:

``` text
scripts/
```

and:

``` text
model/
```

with the Python scripts locating the model using repository-relative
paths.

------------------------------------------------------------------------

# ROS 2 Projects

ROS-based work is normally organized as a ROS workspace:

``` text
ros_ws/
├── src/
├── build/
├── install/
└── log/
```

The normal workflow is:

``` bash
source /opt/ros/humble/setup.bash
cd <path-to-ros_ws>
colcon build
source install/setup.bash
```

Then run the relevant ROS package/node according to that lab or
homework's README.

For example, HW01's ROS implementation uses:

``` bash
ros2 run hw01_tf_demo tf_broadcaster_node
```

and RViz2 for visualization.

------------------------------------------------------------------------

# Generated ROS Directories

ROS workspaces commonly generate:

``` text
build/
install/
log/
```

These are generated by `colcon build`.

They are not manually authored source files.

When preparing the repository for GitHub, these generated directories
should normally be excluded using `.gitignore`, while the actual ROS
source under:

``` text
ros_ws/src/
```

should be retained.

------------------------------------------------------------------------

# Videos and Demonstrations

Where an assignment requires a demonstration, the relevant video is
stored inside that assignment's directory.

For example:

``` text
Homework/hw01-foundations-and-rotations/video-submissions/
```

contains the HW01 demonstration videos.

Similarly, the lab directories contain their own demonstration material
where applicable.

The individual README files provide the corresponding video links and
descriptions when available.

------------------------------------------------------------------------

# AI Use and Academic Documentation

Some course assignments include specific AI-use requirements.

Where AI-use documentation is required, it is stored with the relevant
assignment rather than at the root of the repository.

For example, HW01 contains:

``` text
ai-usenote/
├── Q7.txt
├── Q8.txt
└── Q9.txt
```

The original assignment's AI policy and the corresponding disclosure
requirements should always be followed.

If a future assignment requires an AI-use note, it should be kept inside
that assignment's directory.

------------------------------------------------------------------------

# Adding New Homework

When adding a new homework, use the following general structure:

``` text
Homework/
└── hwXX-name/
    ├── README.md
    ├── code/
    ├── model/
    ├── results/
    └── video-submissions/
```

The exact folders should depend on the assignment.

The important requirement is:

> **Every new homework should have its own README.md.**

The README should explain:

-   what the homework is about
-   what each major file/folder contains
-   how to install dependencies
-   how to run the work
-   where the submitted solutions are
-   where demonstrations/results are stored
-   any special setup requirements

Do not modify the main repository README just because a new homework is
added unless the new homework changes the top-level repository
organization.

------------------------------------------------------------------------

# Adding New Labs

When adding a new lab:

``` text
Lab/
└── labXX-name/
    ├── README.md
    ├── code/
    ├── robot-descriptions/
    ├── results/
    └── video-demo/
```

Again, the exact structure can vary.

The lab's README should be the main source of instructions for that lab.

------------------------------------------------------------------------

# Finding Things Quickly

  ------------------------------------------------------------------------------------
  Looking for...                      Go to...
  ----------------------------------- ------------------------------------------------
  Homework                            `Homework/`

  Labs                                `Lab/`

  Robot models                        `Robot-Descriptions/`

  Textbook/reference                  `Textbook/`

  HW00                                `Homework/hw00-interactive-3d-frame/`

  HW01                                `Homework/hw01-foundations-and-rotations/`

  Lab 1                               `Lab/lab01-mujoco-intro/`

  Lab 2                               `Lab/lab02-fk-heal-franka/`

  Lab 3                               `Lab/lab03-ik-pick-and-place-heal-franka/`

  MuJoCo Menagerie                    `Robot-Descriptions/mujoco_menagerie/`

  Robotis MuJoCo resources            `Robot-Descriptions/robotis_mujoco_menagerie/`

  Spong textbook                      `Textbook/Spong-RobotmodelingandControl.pdf`
  ------------------------------------------------------------------------------------

------------------------------------------------------------------------

# Reproducibility

The repository is intended to be usable by someone other than the
original author.

When running an assignment:

1.  Clone or download the repository.
2.  Navigate to the relevant homework/lab.
3.  Read that directory's `README.md`.
4.  Install only the dependencies required by that activity.
5.  Follow the provided run commands.
6.  Keep generated files and environment-specific files separate from
    source code.

The individual assignment README should always take priority over
generic commands in this root README when the two differ.

------------------------------------------------------------------------

# Repository Conventions

### Directory names

Use descriptive names:

``` text
hw01-foundations-and-rotations
lab02-fk-heal-franka
```

rather than ambiguous names.

### Code

Keep source code close to the assignment that uses it unless it is
genuinely shared across multiple activities.

### Robot descriptions

Put reusable robot descriptions under:

``` text
Robot-Descriptions/
```

and assignment-specific descriptions inside the corresponding assignment
when appropriate.

### Results

Keep generated results such as plots, CSV files, and videos with the
assignment that generated them.

### Documentation

Every major assignment should have a local README.

------------------------------------------------------------------------

# Current Repository Contents

At the current stage, the repository contains:

### Homework

-   **HW00 --- Interactive 3D Frame Rotation Visualizer**
-   **HW01 --- Foundations & Rotations**

### Labs

-   **Lab 01 --- MuJoCo Introduction**
-   **Lab 02 --- Forward Kinematics: HEAL & Franka**
-   **Lab 03 --- IK Pick-and-Place: HEAL & Franka**

### Shared Resources

-   MuJoCo Menagerie resources
-   Robotis MuJoCo Menagerie resources
-   Spong's *Robot Modeling and Control* textbook

As the course progresses, additional material can be added under the
appropriate top-level directory without changing the overall repository
structure.

------------------------------------------------------------------------

# Recommended Navigation for a New User

If you are opening this repository for the first time:

### Step 1

Read this README to understand the repository structure.

### Step 2

Choose what you need:

``` text
Homework/
Lab/
Robot-Descriptions/
Textbook/
```

### Step 3

Open the README inside the specific homework/lab.

### Step 4

Follow that README's setup and execution instructions.

### Step 5

Only install dependencies required for the specific activity you want to
run.

This avoids unnecessary setup for unrelated course work.

------------------------------------------------------------------------

# Course Information

**Course:** ME 639 -- Introduction to Robotics\
**Institution:** Indian Institute of Technology Gandhinagar\
**Academic Year:** 2026--27

------------------------------------------------------------------------

# Author

**Kshitij Suresh Giri**

B.Tech Dual Major --- Mechanical Engineering & Artificial Intelligence\
Indian Institute of Technology Gandhinagar

------------------------------------------------------------------------

## Repository Purpose

This repository is intended to serve as a single, organized location for
the course's:

-   homework
-   laboratory work
-   code
-   robot models
-   simulations
-   visualizations
-   results
-   demonstrations
-   supporting references

The repository is deliberately structured so that **individual
assignment folders remain self-contained**, while the root README
provides only the high-level map.

For detailed instructions, always move from:

``` text
README.md
    ↓
specific Homework/Lab folder
    ↓
specific Homework/Lab README.md
    ↓
code / models / results / demonstrations
```

That structure is intended to remain stable even as new course material
is added.