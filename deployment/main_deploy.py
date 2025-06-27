import sys
sys.path.append("../")

import numpy as np
import cv2
import math
import time
from PIL import Image

from v4.yolo import YOLO
from camutils import get_world_pos, get_camera, get_camera_param, transfer_pos
from roboutils import setup_abb_egm_client, setup_pybullet, get_joints, send_joints, clip
from velocitymodel import BallVelocityEstimator, BallTrajectoryPredictor

import torch

sys.path.append("../")
from impact.model import RobotActionModel


if __name__ == '__main__':
    render = True
    enable_robot = False

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
    features = {'position': [],
                'velocity': [],
                'end_position': [],
                'target_joint': []}

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

    mymodel = RobotActionModel()
    checkpoint = torch.load('./checkpoints/model.pth', map_location='cpu')
    mymodel.load_state_dict(checkpoint['model_state_dict'])
    mymodel.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mymodel.to(device)


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
                    features = {'position': [],
                                'velocity': [],
                                'end_position': [],
                                'target_joint': []}


                bp_list_yolo.append(ball_pos_yolo)
                
                current_time = time.time()
                position = ball_pos_yolo
                
                # Add position data to velocity estimator
                velocity_estimator_yolo.add_position(current_time, position)
                
                # Estimate velocity
                filtered_pos, filtered_vel = velocity_estimator_yolo.estimate_velocity()
                
                     
                if filtered_pos is not None and filtered_vel is not None and filtered_vel[1] < 0:
                    features['position'].append(filtered_pos)
                    features['velocity'].append(filtered_vel)
                    # Perform trajectory prediction
                    prediction = trajectory_predictor_yolo.predict_trajectory(
                        current_time, filtered_pos, filtered_vel)
                    
                    # Output prediction results
                    if prediction['hittable']:
                        print('****************** yolo ******************')
                        print("可击打！预测位置 world:", prediction['end_position'])
                        print("预测速度:", prediction['end_velocity'])
                        print("预测时间:", prediction['end_time'])
                        pred_pos_base = np.array(transfer_pos(prediction['end_position']))
                        print("可击打！预测位置 base:", pred_pos_base)
                        print('****************** yolo ******************')

                        features['end_position'].append(prediction['end_position'])
                        
                        
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
                            
                            joint_positions = p.calculateInverseKinematics(
                                bodyUniqueId = robot_id,
                                endEffectorLinkIndex = 6,
                                targetPosition = (pred_pos_base[0], pred_pos_base[1], pred_pos_base[2]+0.15),
                                targetOrientation= (-0.001, 0.999, -0.001, -0.001),
                                maxNumIterations=30
                            )

                            target_joint = [math.degrees(item) for item in joint_positions]
                            target_joint = clip(target_joint)
                            print('***************************************************')
                            print(f"Computed joints: {target_joint}")
                            print('***************************************************')

                            features['target_joint'].append(target_joint)

                            if enable_robot and send_cnt == 0:
                                send_joints(egm_client, target_joint)
                                time.sleep(0.15)
                                
                                # Update current_joints after sending command
                                current_joints = target_joint
                                
                                # Calculate hit joint
                                
                                for key in features:
                                    if key == 'position':
                                        filtered_pos = np.array(features[key])

                                        if filtered_pos.shape[0] > 3:
                                            filtered_pos = filtered_pos[:3]
                                        elif filtered_pos.shape[0] == 2:
                                            mid_pos = (filtered_pos[0] + filtered_pos[1]) / 2
                                            filtered_pos = np.vstack((filtered_pos[0], mid_pos, filtered_pos[1]))
                                        elif filtered_pos.shape[0] == 1:
                                            filtered_pos = np.vstack((filtered_pos[0], filtered_pos[0], filtered_pos[0]))

                                        features[key] = torch.tensor(filtered_pos.reshape(1, -1), dtype=torch.float32).to(device)
                                    elif key == 'velocity':
                                        filtered_vel = np.array(features[key])

                                        if filtered_vel.shape[0] > 3:
                                            filtered_vel = filtered_vel[:3]
                                        elif filtered_vel.shape[0] == 2:
                                            mid_vel = (filtered_vel[0] + filtered_vel[1]) / 2
                                            filtered_vel = np.vstack((filtered_vel[0], mid_vel, filtered_vel[1]))
                                        elif filtered_vel.shape[0] == 1:
                                            filtered_vel = np.vstack((filtered_vel[0], filtered_vel[0], filtered_vel[0]))

                                        features[key] = torch.tensor(filtered_vel.reshape(1, -1), dtype=torch.float32).to(device)
                                    else:
                                        features[key] = torch.tensor(features[key], dtype=torch.float32).to(device)

                                with torch.no_grad():
                                    pred = mymodel(features)
                                    pred = pred.squeeze().cpu().numpy()
                                
                                print('Prediction', pred)
                                axis2_change, axis4_change, axis5_change = pred

                                features = {'position': [],
                                        'velocity': [],
                                        'end_position': [],
                                        'target_joint': []}

                                hit_joint = target_joint.copy()
                                hit_joint[2] += axis2_change  
                                hit_joint[4] += axis4_change  
                                hit_joint[5] += axis5_change  
                                hit_joint = clip(hit_joint)                 
                                
                                # Send hit joint command
                                send_joints(egm_client, hit_joint)
                                time.sleep(0.5)
                                
                                # Return to initial position
                                send_joints(egm_client, init_joints)
                                
                                send_cnt = 1        
                    

    except OSError as e:
        print(f"Error while receiving data: {e}")
    finally:
        # Clean up
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Closing connection.")