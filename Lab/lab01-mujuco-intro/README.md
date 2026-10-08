# Lab 1 — MuJoCo Introduction

**Course:** ME639 — Introduction to Robotics  
**Author:** Kshitij Suresh Giri  
**Roll No.:** 23110177  
**Program:** Dual Major B.Tech (Mechanical Engineering & Artificial Intelligence)  
**Institute:** IIT Gandhinagar

---

## 📌 Overview

This lab introduces basic robot simulation and interaction using **MuJoCo**.

The lab contains two challenges:

- **Challenge 1:** TurtleBot3 Waffle Pi
- **Challenge 2:** Quadrotor

The simulations allow keyboard-based control, visualization of the robot body frame, and display of the robot position and rotation matrix.

---

## 📋 Lab Tasks

### Challenge 1 — TurtleBot3 Waffle Pi

The TurtleBot3 Waffle Pi is spawned in MuJoCo.

The implementation:

1. Spawns the TurtleBot3 Waffle Pi.
2. Uses the keyboard to move the robot.
3. Shows how the robot body frame changes as the robot moves.
4. Displays the robot's position and rotation matrix.

### Challenge 2 — Quadrotor

A quadrotor is spawned in MuJoCo.

The implementation repeats the relevant visualization and control steps from Challenge 1:

1. Spawn a quadrotor.
2. Control it using the keyboard.
3. Show how the body frame changes.
4. Display the position and rotation matrix.

---

## 📁 Folder Structure

```text
lab01-mujoco-intro/
│
├── code/
│   ├── quadrotor_kb.py
│   └── tb3_waffle_kb.py
│
├── explanation/
│   └── jitter_expl.txt
│
├── robot-descriptions/
│   ├── Quadrotor/
│   │   ├── quadrator.xml
│   │   └── scene.xml
│   │
│   └── robotis_tb3/
│       ├── assets/
│       ├── LICENSE
│       ├── scene_turtlebot3_burger.xml
│       ├── scene_turtlebot3_waffle_pi.xml
│       ├── tb3_burger.png
│       ├── tb3_waffle_pi.png
│       ├── turtlebot3_burger.xml
│       └── turtlebot3_waffle_pi.xml
│
├── video-demo/
│   ├── lab1-quadrotor.mp4
│   └── lab1-tb3.mp4
│
└── README.md
```

---

## ▶️ How to Run

From the repository root:

### TurtleBot3 Waffle Pi

```bash
python3 Lab/lab01-mujoco-intro/code/tb3_waffle_kb.py
```

### Quadrotor

```bash
python3 Lab/lab01-mujoco-intro/code/quadrotor_kb.py
```

The scripts use **repository-relative paths**, so no computer-specific absolute paths are required.

---

## 🎮 Keyboard Controls

### TurtleBot3

- **↑** — Move forward
- **↓** — Move backward
- **←** — Turn left
- **→** — Turn right
- **Space** — Stop

### Quadrotor

- **↑** — Hold to Move Forward
- **↓** — Hold to Move Backward
- **←** — Hold to Move left
- **→** — Hold to Move right
- **Space** — Hold for Upthrust, Release to hover
- **Shift** — Hold for Downthrust, Release to hover

---

## 🎥 Video Demonstrations

### Quadrotor

[▶️ Watch Quadrotor Demonstration on YouTube](https://youtu.be/0KQISTDIv3Y)

A local copy is also available:

```text
video-demo/lab1-quadrotor.mp4
```

### TurtleBot3 Waffle Pi

[▶️ Watch TurtleBot3 Demonstration on YouTube](https://youtu.be/xLH_huWhrXI)

A local copy is also available:

```text
video-demo/lab1-tb3.mp4
```

---

## 🛠️ Technologies Used

- Python
- NumPy
- GLFW
- MuJoCo
- MuJoCo Viewer

---

## 👨‍💻 Author

**Kshitij Suresh Giri**  
Roll No. 23110177  
Dual Major B.Tech (Mechanical Engineering & Artificial Intelligence)  
IIT Gandhinagar