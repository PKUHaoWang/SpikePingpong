from abb_egm_pyclient.egm_client import EGMClient
from abb_egm_pyclient import DEFAULT_UDP_PORT
from typing import List, Sequence
import pybullet as p
import pybullet_data
import numpy as np
import math


def setup_abb_egm_client():
    port = DEFAULT_UDP_PORT
    egm_client = EGMClient(port=port)  
    init_joint = get_joints(egm_client)
    send_joints(egm_client, init_joint)
    return egm_client

def setup_pybullet():
    p.connect(p.GUI)
    p.setTimeStep(1/240)
    #p.setTimeStep(20)
    p.setGravity(0, 0, -9.8) 
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    urdf_path = "./PyBullet_Industrial_Robotics_Gym/URDFs/Robots/ABB_IRB_120/ABB_IRB_120.urdf"  # 你的机器人路径
    robot_id = p.loadURDF(urdf_path, useFixedBase=True)
    return p, robot_id

def get_joints(egm_client):
    pb_robot_msg = egm_client.receive_msg()
    cur_joints: Sequence[float] = pb_robot_msg.feedBack.joints.joints
    return cur_joints


def send_joints(egm_client, target_conf: Sequence[float]):
    pb_robot_msg = egm_client.receive_msg()
    start_conf: List[float] = pb_robot_msg.feedBack.joints.joints

    if len(start_conf) == len(target_conf) - 1:
        start_conf.append(pb_robot_msg.feedBack.externalJoints.joints[0])

    start_conf_arr = np.array(start_conf)
    target_conf_arr = np.array(target_conf)

    deltas = target_conf_arr - start_conf_arr

    num_msgs = 25
    for n in range(num_msgs):
        conf = []
        for start_pos, delta in zip(start_conf, deltas):
            abs_change = n * delta / num_msgs
            conf.append(start_pos + abs_change)

        egm_client.send_planned_configuration(conf)

def clip(action):
    joint_range = [(-1.57, 1.57),(-0.91986, 0.91986),(-0.91986, 0.919863),(-2.79253, 2.79253),(-2.0944, 2.0944),(-1.63, 1.63)]
    for i in range(6):
        jmin, jmax = joint_range[i]
        jmin = jmin / math.pi * 180
        jmax = jmax / math.pi * 180
        if action[i] <= jmin:
            action[i] = jmin + 2
        elif action[i] >= jmax:
            action[i] = jmax - 2
    return action