import time
from pathlib import Path

import numpy as np
import glfw
import mujoco
import mujoco.viewer
import warnings


warnings.filterwarnings(
    "ignore",
    category=glfw.GLFWError,
    message=".*Wayland: The platform does not provide the window position.*"
)


# ------------------------------------------------
# PATHS
# ------------------------------------------------

# Get the Lab 1 directory
LAB_DIR = Path(__file__).resolve().parent.parent

# Path to the TurtleBot3 Waffle Pi MuJoCo model
XML_PATH = (
    LAB_DIR
    / "robot-descriptions"
    / "robotis_tb3"
    / "scene_turtlebot3_waffle_pi.xml"
)


# Load the TurtleBot3 Waffle Pi model
model = mujoco.MjModel.from_xml_path(str(XML_PATH))

data = mujoco.MjData(model)


# Find the two wheel motors
left = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_left"
)

right = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wheel_right"
)

# Find the robot's base body
base = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "base"
)


# Current keyboard command
command = "stop"


# This function is called whenever a key is pressed
def keyboard(key):

    global command

    if key == glfw.KEY_UP:
        command = "forward"

    elif key == glfw.KEY_DOWN:
        command = "backward"

    elif key == glfw.KEY_LEFT:
        command = "left"

    elif key == glfw.KEY_RIGHT:
        command = "right"

    elif key == glfw.KEY_SPACE:
        command = "stop"


# Draw one arrow from start to end
def draw_arrow(viewer, number, start, end, color):

    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[number],
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.array([0.015, 0.015, 0.015]),
        start,
        np.eye(3).flatten(),
        color
    )

    mujoco.mjv_connector(
        viewer.user_scn.geoms[number],
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.015,
        start,
        end
    )


# Open the MuJoCo window
with mujoco.viewer.launch_passive( model, data, key_callback=keyboard) as viewer:

    while viewer.is_running():

        # ------------------------------------------------
        # ROBOT CONTROL
        # ------------------------------------------------

        speed = 7.0

        if command == "forward":
            data.ctrl[left] = speed
            data.ctrl[right] = speed

        elif command == "backward":
            data.ctrl[left] = -speed
            data.ctrl[right] = -speed

        elif command == "left":
            data.ctrl[left] = -speed
            data.ctrl[right] = speed

        elif command == "right":
            data.ctrl[left] = speed
            data.ctrl[right] = -speed

        elif command == "stop":
            data.ctrl[left] = 0
            data.ctrl[right] = 0

        # Move the simulation one step forward
        mujoco.mj_step(model, data)


        # ------------------------------------------------
        # GET BODY POSITION AND ROTATION
        # ------------------------------------------------

        position = data.xpos[base]

        x = position[0]
        y = position[1]
        z = position[2]

        # Rotation matrix of robot body with respect to the world/inertial frame
        R = data.xmat[base].reshape(3, 3)
        R[np.abs(R) < 0.005] = 0

        # ------------------------------------------------
        # BODY FRAME
        # ------------------------------------------------

        # Put the body-frame origin slightly above the robot so that we can see the axes
        body_origin = position + np.array([0, 0, 0.13])

        axis_length = 0.30

        # Body x, y and z axes
        body_x = body_origin + R[:, 0] * axis_length
        body_y = body_origin + R[:, 1] * axis_length
        body_z = body_origin + R[:, 2] * axis_length


        # ------------------------------------------------
        # DRAW BODY FRAME
        # ------------------------------------------------

        with viewer.lock():

            viewer.user_scn.ngeom = 0

            # Body frame
            draw_arrow(
                viewer, 0, body_origin, body_x,
                np.array([1, 0, 0, 1])
            )

            draw_arrow(
                viewer, 1, body_origin, body_y,
                np.array([0, 1, 0, 1])
            )

            draw_arrow(
                viewer, 2, body_origin, body_z,
                np.array([0, 0, 1, 1])
            )

            viewer.user_scn.ngeom = 6


        # ------------------------------------------------
        # DISPLAY POSITION AND ROTATION MATRIX
        # ------------------------------------------------

        text = (
            f"BODY FRAME\n"
            f"x = {x:.3f} m\n"
            f"y = {y:.3f} m\n"
            f"z = {z:.3f} m\n\n"
            f"ROTATION MATRIX R\n"
            f"[{R[0,0]:.3f}  {R[0,1]:.3f}  {R[0,2]:.3f}]\n"
            f"[{R[1,0]:.3f}  {R[1,1]:.3f}  {R[1,2]:.3f}]\n"
            f"[{R[2,0]:.3f}  {R[2,1]:.3f}  {R[2,2]:.3f}]"
        )

        viewer.set_texts([
            (
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                text,
                ""
            )
        ])

        # Update the MuJoCo window
        viewer.sync()

        time.sleep(model.opt.timestep)