# Lab 2 — Forward Kinematics: HEAL & Franka

**Course:** ME639 — Introduction to Robotics  
**Author:** Kshitij Suresh Giri  
**Roll No.:** 23110177  
**Program:** Dual Major B.Tech (Mechanical Engineering & Artificial Intelligence)  
**Institute:** IIT Gandhinagar

---

## 📌 Overview

This lab focuses on **Forward Kinematics (FK)** of robotic manipulators using analytical **Denavit–Hartenberg (DH)** transformations and **MuJoCo** simulation.

The lab covers:

- A 2-DOF spatial manipulator
- 6-DOF HEAL robotic arm
- 7-DOF Franka robotic arm
- Analytical FK using DH parameters
- FK verification using MuJoCo
- Comparison of analytical and simulation-based results

The robot-description files are included inside the repository so that the code can be run without computer-specific absolute paths.

---

## 📋 Lab Tasks

### Task 1 — 2-DOF Spatial Manipulator

A 2-DOF spatial manipulator is analyzed using transformation matrices.

The solution includes the required forward-kinematics formulation and results.

**Solution:**
```text
task1/task1-soln.pdf
```

---

### Task 2 — Analytical Forward Kinematics Using DH Parameters

Forward kinematics is implemented using the **Denavit–Hartenberg (DH) convention** for:

- **HEAL — 6 DOF**
- **Franka — 7 DOF**

The scripts construct the individual DH transformation matrices and calculate the final end-effector transformation.

**Files:**
```text
task2/heal_FK_DH.py
task2/franka_FK_DH.py
```

---

### Task 3 — Forward Kinematics Using MuJoCo

The same robots are loaded into MuJoCo and their end-effector poses are obtained directly from the simulation model.

This provides a way to verify the analytical FK results against the robot simulation.

**Files:**
```text
task3/heal_FK_mujoco.py
task3/franka_FK_mujoco.py
```

---

### Comparison — Task 2 vs Task 3

The analytical DH-based FK results are compared with the MuJoCo FK results for both robots.

**Files:**
```text
comparison-task2-3/heal_FK.py
comparison-task2-3/franka_FK.py
```

These scripts use repository-relative paths, so the repository can be cloned or forked and run without changing PC-specific file paths.

---

## 📁 Folder Structure

```text
lab02-fk-heal-franka/
│
├── comparison-task2-3/
│   ├── franka_FK.py
│   └── heal_FK.py
│
├── robot-descriptions/
│   ├── franka/
│   │   └── mjx_panda.xml
│   │
│   └── single_arm_heal_effort_actuation_rs_mj_2.xml
│
├── task1/
│   └── task1-soln.pdf
│
├── task2/
│   ├── franka_FK_DH.py
│   └── heal_FK_DH.py
│
├── task3/
│   ├── franka_FK_mujoco.py
│   └── heal_FK_mujoco.py
│
├── video-demo/
│   └── Lab2.mp4
│
└── README.md
```

---

## ▶️ How to Run

Make sure the required Python packages, including **MuJoCo** and **NumPy**, are installed.

From the repository root:

### Task 2 — Analytical DH FK

```bash
python3 Lab/lab02-fk-heal-franka/task2/franka_FK_DH.py
python3 Lab/lab02-fk-heal-franka/task2/heal_FK_DH.py
```

### Task 3 — MuJoCo FK

```bash
python3 Lab/lab02-fk-heal-franka/task3/franka_FK_mujoco.py
python3 Lab/lab02-fk-heal-franka/task3/heal_FK_mujoco.py
```

### Comparison

```bash
python3 Lab/lab02-fk-heal-franka/comparison-task2-3/franka_FK.py
python3 Lab/lab02-fk-heal-franka/comparison-task2-3/heal_FK.py
```

You can also run the scripts from inside their respective directories.

---

## 🤖 Robot Descriptions

The required MuJoCo robot models are stored locally in:

```text
robot-descriptions/
```

### Franka

```text
robot-descriptions/franka/mjx_panda.xml
```

### HEAL

```text
robot-descriptions/single_arm_heal_effort_actuation_rs_mj_2.xml
```

The Python scripts resolve these paths relative to the repository structure, avoiding hard-coded paths such as:

```text
/home/<username>/...
```

This makes the lab easier to clone, fork, and run on another computer.

---

## 🔍 What Is Verified

The lab verifies forward kinematics through two approaches:

### 1. Analytical FK

The robot's DH parameters are used to calculate homogeneous transformation matrices:

```text
T = A1 A2 ... An
```

The final transformation gives the end-effector position and orientation.

### 2. MuJoCo FK

The same robot configuration is loaded into MuJoCo, and the end-effector pose is obtained from the simulation.

### 3. Comparison

The analytical and MuJoCo results are compared to check that the FK calculations are consistent.

---

## 🎥 Video Demonstration

A complete video demonstration of Lab 2 is included in the repository:

```text
video-demo/Lab2.mp4
```

### YouTube Demonstration

[▶️ Watch Lab 2 Video Demonstration on YouTube](https://youtu.be/aLbu2stbzhA)

The video demonstrates the completed Lab 2 work, including the forward-kinematics implementations and MuJoCo simulations.

---

## 🛠️ Technologies Used

- Python
- NumPy
- MuJoCo
- Denavit–Hartenberg (DH) transformations
- MuJoCo robot models

---

## 📌 Recommended Navigation

If you are reviewing this lab for the first time, the recommended order is:

1. **Task 1** — Understand the basic spatial manipulator problem.
2. **Task 2** — Study the analytical DH-based FK implementations.
3. **Task 3** — Run the corresponding MuJoCo FK implementations.
4. **Comparison** — Compare the analytical and simulation results.
5. **Video Demo** — Watch the complete demonstration.

---

## 👨‍💻 Author

**Kshitij Suresh Giri**  
Roll No. 23110177  
Dual Major B.Tech (Mechanical Engineering & Artificial Intelligence)  
IIT Gandhinagar