import sys
sys.path.append("../")

import numpy as np
import cv2
import math
import time
import os
import json
import random
from datetime import datetime
from PIL import Image

from deployment.v4.yolo import YOLO
from deployment.camutils import get_world_pos, get_camera, get_camera_param, get_xy_from_hsv, transfer_pos
from deployment.roboutils import setup_abb_egm_client, setup_pybullet, get_joints, send_joints, clip
from deployment.velocitymodel import BallVelocityEstimator, BallTrajectoryPredictor

from imitation.saveutils import ensure_dir, CustomEncoder, save_data




if __name__ == '__main__':
    render = True
    enable_robot = True
    enable_random_hit = True  # 新增标志位，控制是否使用随机角度

    # Create data directory
    data_dir = "./data"
    cur_time = datetime.now().strftime("%Y%m%d_%H%M%S.%f")
    data_dir = os.path.join(data_dir, cur_time)
    ensure_dir(data_dir)

    ## Set up camera
    pipeline, align, device_ids = get_camera()
    intr, extr, depth_scale = get_camera_param(device_ids)

    ## Yolo model
    yolo = YOLO()
    
    ## Other variables
    cnt = 0
    send_cnt = 0
    target_joint = None
    frame_num = 0

    # Episode tracking
    current_episode_dir = None
    is_new_episode = True

    # ball_pos = None
    ball_pos_yolo = None
    correct_pos = None
    bp_list_yolo = []
    history = []
    velocity_estimator_yolo = BallVelocityEstimator()
    trajectory_predictor_yolo = BallTrajectoryPredictor()

    if enable_robot:
        egm_client = setup_abb_egm_client()
        init_joints = get_joints(egm_client)
        init_joints = np.array(init_joints)
    else:
        init_joints = None
    
    p, robot_id = setup_pybullet()

    filtered_pos = None
    filtered_vel = None
    current_joints = init_joints

    try:
        while True:
            # **************** get color image and depth image ****************
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue
            
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            if color_image.shape[2] == 3:
                color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            color_image = color_image.astype(np.uint8)
            depth_image = depth_image.astype(np.float32) * depth_scale

            if render:
                color_image_render = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                cv2.imshow('L455 Image', color_image_render)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # **************** yolo detect ****************
            height, width = color_image.shape[:2]
            mask_height = int(height * 1/5)
            masked_image = color_image.copy()
            masked_image[0:mask_height, :, :] = 0

            image1 = Image.fromarray(masked_image) 
            r_image1, center_x_1, center_y_1 = yolo.detect_image(image1, crop = False, count = False)
            if r_image1 == -1:
                continue
            r_image1 = np.array(r_image1)
            r_image1 = cv2.cvtColor(r_image1, cv2.COLOR_RGB2BGR)
            
            if render:
                cv2.imshow('Color Image', r_image1)
                cnt += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if center_y_1 is not None:
                ball_pos_yolo = get_world_pos(depth_image, center_x_1, center_y_1, intr, extr)
                if ball_pos_yolo is not None:
                    ball_pos_base = transfer_pos(ball_pos_yolo)
                    print('world:', ball_pos_yolo, 'base:', ball_pos_base)

            if ball_pos_yolo is not None:
                # Check if this is a new episode (ball goes to the other side of the table)
                if ball_pos_yolo[1] > -0.5:
                    print('&&&&&&&&&&&&&&&&&&&&&&&&&& New Episode &&&&&&&&&&&&&&&&&&&&&&&&&&')
                    send_cnt = 0
                    history = []
                    target_joint = None
                    frame_num = 0
                    
                    # Create a new episode directory with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    current_episode_dir = os.path.join(data_dir, timestamp)
                    ensure_dir(current_episode_dir)
                    print(f"Created new episode directory: {current_episode_dir}")
                    
                    # Create subdirectories for images and robot data
                    l455_dir = os.path.join(current_episode_dir, "l455")
                    robot_dir = os.path.join(current_episode_dir, "robot")
                    
                    ensure_dir(l455_dir)
                    ensure_dir(robot_dir)
                    
                    # Initialize trajectory file with empty frames array
                    trajectory_file = os.path.join(current_episode_dir, "trajectory.json")
                    with open(trajectory_file, 'w') as f:
                        json.dump({"frames": []}, f, indent=2)
                    
                    # Save episode metadata
                    metadata = {
                        "timestamp": timestamp,
                        "camera_intrinsics": intr,
                        "camera_extrinsics": extr,
                        "depth_scale": depth_scale
                    }
                    with open(os.path.join(current_episode_dir, "metadata.json"), 'w') as f:
                        json.dump(metadata, f, cls=CustomEncoder, indent=2)
                    
                    is_new_episode = True
                
                if current_episode_dir is None:
                    # If we have ball data but no episode directory yet, create one
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    current_episode_dir = os.path.join(data_dir, timestamp)
                    ensure_dir(current_episode_dir)
                    print(f"Created new episode directory: {current_episode_dir}")

                bp_list_yolo.append(ball_pos_yolo)
                
                current_time = time.time()
                position = ball_pos_yolo
                
                # Add position data to velocity estimator
                velocity_estimator_yolo.add_position(current_time, position)
                
                # Estimate velocity
                filtered_pos, filtered_vel = velocity_estimator_yolo.estimate_velocity()
                
                # Save this frame of data if we have a valid episode directory
                if current_episode_dir is not None:
                    # Get current robot joints if available
                    if enable_robot:
                        current_joints = get_joints(egm_client)
                    
                    # Save data for this frame
                    save_data(
                        current_episode_dir, 
                        frame_num, 
                        color_image, 
                        ball_pos_yolo, 
                        filtered_pos, 
                        filtered_vel, 
                        current_joints
                    )
                    frame_num += 1
                
                if filtered_pos is not None and filtered_vel is not None and filtered_vel[1] < 0:
                    # Perform trajectory prediction
                    prediction = trajectory_predictor_yolo.predict_trajectory(
                        current_time, filtered_pos, filtered_vel)
                    
                    # Output prediction results
                    if prediction['hittable']:
                        print('****************** yolo ******************')
                        print("可击打！预测位置 world:", prediction['end_position'])
                        print("预测速度:", prediction['end_velocity'])
                        print("预测时间:", prediction['end_time'])
                        if -0.1 < prediction['end_position'][0] < 0.05:
                            prediction['end_position'][0] -= 0.08
                            prediction['end_position'][2] -= 0.03
                        elif prediction['end_position'][0] < -0.1:
                            prediction['end_position'][0] -= 0.12
                        pred_pos_base = np.array(transfer_pos(prediction['end_position']))

                        print("可击打！预测位置 base:", pred_pos_base)
                        print('****************** yolo ******************')

                        # Save prediction data to main directory
                        if current_episode_dir is not None:
                            prediction_data = {
                                "frame": frame_num,
                                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S.%f"),
                                "prediction": prediction,
                                "pred_pos_base": pred_pos_base
                            }
                            
                            prediction_file = os.path.join(current_episode_dir, "predictions.json")
                            try:
                                if os.path.exists(prediction_file):
                                    with open(prediction_file, 'r') as f:
                                        all_predictions = json.load(f)
                                else:
                                    all_predictions = {"predictions": []}
                                
                                # Add new prediction data
                                all_predictions["predictions"].append(prediction_data)
                                
                                # Save updated predictions
                                with open(prediction_file, 'w') as f:
                                    json.dump(all_predictions, f, cls=CustomEncoder, indent=2)
                                    
                            except Exception as e:
                                print(f"Error updating predictions file: {e}")
                                # Fallback: save individual prediction
                                with open(os.path.join(current_episode_dir, f"prediction_{frame_num:04d}.json"), 'w') as f:
                                    json.dump(prediction_data, f, cls=CustomEncoder, indent=2)
                        
                        if pred_pos_base[1] < -0.5:
                            pred_pos_base[1] = -0.49
                        if pred_pos_base[1] > 0.5:
                            pred_pos_base[1] = 0.49
                        if pred_pos_base[2] < 0.25:
                            pred_pos_base[2] = 0.26
                        if pred_pos_base[2] > 0.7:
                            pred_pos_base[2] = 0.69

                        y_flag = pred_pos_base[1] >= -0.5 and pred_pos_base[1] <= 0.5
                        z_flag = pred_pos_base[2] >= 0.25 and pred_pos_base[2] <= 0.7
                        if y_flag and z_flag:
                            print('***************************************************')
                            joint_positions = p.calculateInverseKinematics(
                                bodyUniqueId = robot_id,
                                endEffectorLinkIndex = 6,
                                targetPosition = (pred_pos_base[0], pred_pos_base[1], pred_pos_base[2]+0.15),
                                targetOrientation= (-0.001, 0.999, -0.001, -0.001),
                                maxNumIterations=30
                            )

                            target_joint = [math.degrees(item) for item in joint_positions]
                            target_joint = clip(target_joint)
                            print(f"Computed joints: {target_joint}")
                            print('***************************************************')
                            
                            # Prepare to save robot joint data
                            if current_episode_dir is not None:
                                robot_dir = os.path.join(current_episode_dir, "robot")
                                ensure_dir(robot_dir)
                                
                                # Save target joints
                                robot_data = {
                                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S.%f"),
                                    "target_joint": target_joint
                                }

                            if enable_robot and send_cnt == 0:
                                send_joints(egm_client, target_joint)
                                time.sleep(0.15)
                                
                                # Update current_joints after sending command
                                current_joints = target_joint
                                
                                # Get actual joints after sending target joint command
                                if current_episode_dir is not None:
                                    actual_target_joints = get_joints(egm_client)
                                    robot_data["actual_target_joint"] = actual_target_joints
                                
                                # 计算随机或固定的hit joint角度变化
                                if enable_random_hit:
                                    # 根据要求生成随机角度变化
                                    axis2_change = 15.0 + random.uniform(-10.0, 10.0)  # 15 ± 10度
                                    axis4_change = 60.0 + random.uniform(-10.0, 10.0)  # 70 ± 10度
                                    axis5_change = random.uniform(-20.0, 20.0)  # ±20度
                                    
                                    print(f"Random hit joint changes - Axis 2: {axis2_change:.2f}°, Axis 4: {axis4_change:.2f}°, Axis 5: {axis5_change:.2f}°")
                                else:
                                    # 使用固定值
                                    axis2_change = 15.0
                                    axis4_change = 60.0
                                    axis5_change = 0.0
                                    
                                # 记录角度变化（无论是随机还是固定）
                                hit_joint_changes = {
                                    "axis2_change": axis2_change,
                                    "axis4_change": axis4_change,
                                    "axis5_change": axis5_change,
                                    "is_random": enable_random_hit
                                }
                                
                                # 保存到robot_data中
                                if current_episode_dir is not None:
                                    robot_data["hit_joint_changes"] = hit_joint_changes
                                
                                # Calculate hit joint with random or fixed changes
                                hit_joint = target_joint.copy()
                                hit_joint[2] -= axis2_change  # 第2轴的变化
                                hit_joint[4] -= axis4_change  # 第4轴的变化
                                hit_joint[5] += axis5_change  # 第5轴的变化（新增）
                                hit_joint = clip(hit_joint)
                                
                                # Store hit joint data
                                if current_episode_dir is not None:
                                    robot_data["hit_joint"] = hit_joint
                                
                                # Send hit joint command
                                send_joints(egm_client, hit_joint)
                                time.sleep(0.5)

                                # Get actual joints after hitting
                                if current_episode_dir is not None:
                                    actual_hit_joints = get_joints(egm_client)
                                    robot_data["actual_hit_joint"] = actual_hit_joints
                                
                                # Return to initial position
                                send_joints(egm_client, init_joints)
                                
                                # Get actual joints after returning to initial position
                                if current_episode_dir is not None:
                                    actual_init_joints = get_joints(egm_client)
                                    robot_data["init_joint"] = init_joints.tolist() if isinstance(init_joints, np.ndarray) else init_joints
                                    robot_data["actual_init_joint"] = actual_init_joints
                                    
                                    # Save all robot joint data to a single file
                                    with open(os.path.join(robot_dir, f"joints_{frame_num:04d}.json"), 'w') as f:
                                        json.dump(robot_data, f, cls=CustomEncoder, indent=2)
                                
                                send_cnt = 1        
                    

    except OSError as e:
        print(f"Error while receiving data: {e}")
    finally:
        # Clean up
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Closing connection.")