import numpy as np
import pyrealsense2 as rs
import cv2

def get_camera():
    context = rs.context()
    devices = context.query_devices()
    device_ids = [device.get_info(rs.camera_info.serial_number) for device in devices]

    # assert len(device_ids) == 1 # only one RGB-D camera
    device_id = device_ids[0]

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device_id)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

    pipeline.start(config)

    align = rs.align(rs.stream.depth)
    return pipeline, align, device_ids

def get_camera_l515():
    context = rs.context()
    devices = context.query_devices()

    device_ids = [device.get_info(rs.camera_info.serial_number) for device in devices]

    # assert len(device_ids) == 1 # only one RGB-D camera
    device_id = device_ids[1]

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device_id)
    config.enable_stream(rs.stream.color,960, 540, rs.format.bgr8, 60)

    pipeline.start(config)
    return pipeline

def get_camera_param(device_ids):
    intr = np.load(f'camera_intrin_{str(device_ids[0])}.npy')
    extr = np.load(f'camera_extrin_{str(device_ids[0])}.npy')
    depth_scale = np.load(f'camera_depthscale_{str(device_ids[0])}.npy')
    return intr, extr, depth_scale

def get_world_pos(depth_image, center_x, center_y, intr, extr):
    u = int(center_x)
    v = int(center_y)
    z = depth_image[u, v]

    # **************** get 3d pos ****************
    # to camera coord
    fx = intr[0, 0]
    fy = intr[1, 1]
    ppx = intr[0, 2]
    ppy = intr[1, 2]

    x_cam = (v - ppx) * z / fx
    y_cam = (u - ppy) * z / fy
    point_cam = np.array([x_cam, y_cam, z, 1])
    
    # to world coord
    point_world = np.transpose(extr @ point_cam)
    ball_pos = point_world
    ball_pos = ball_pos[:3]

    # align with simulator world coord
    ball_pos[2] += 0.76
    ball_pos = ball_pos + np.array([0.05, -0.37, -0.01]) 

    if -0.74 < ball_pos[0] < 0.74 and ball_pos[1] > -1.6 and ball_pos[2] < 1.5:
        return ball_pos
    else:
        return None


def transfer_pos(world_pos):
    x, y, z = world_pos
    x_base = y + 1.815 
    y_base = -x
    z_base = z - 0.82 + 0.17
    return float(x_base), float(y_base), float(z_base)


def get_xy_from_hsv(frame):
    lower_orange = np.array([10, 150, 150])
    upper_orange = np.array([30, 255, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 根据颜色范围创建掩膜
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('mask', mask)
    cv2.imshow('mask frame', masked_frame)

    # 查找感兴趣像素
    interest_pixels = cv2.findNonZero(mask)

    if interest_pixels is not None and len(interest_pixels) > 0:
        x_max = max(interest_pixels, key=lambda p: p[0][0])[0][0]
        x_min = min(interest_pixels, key=lambda p: p[0][0])[0][0]
        y_max = max(interest_pixels, key=lambda p: p[0][1])[0][1]
        y_min = min(interest_pixels, key=lambda p: p[0][1])[0][1]

        # print(x_max, x_min, y_max, y_min)

        c_x = x_min + int((x_max - x_min) / 2)
        c_y = y_min + int((y_max - y_min) / 2)

        display_radius = int(x_max - x_min)
        frame = cv2.circle(frame, (c_x, c_y), radius=int(display_radius * 1), color=(0, 255, 0), thickness=int(display_radius / 3))

        return c_x, c_y
    
    return None, None
    
