# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:43:50 2024

@author: Houch
"""

import pyrealsense2 as rs
import numpy as np

# 配置管道
pipeline = rs.pipeline()
config = rs.config()

# 配置流
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
# 启动管道
pipeline.start(config)

try:
    # 等待一帧以获取相机信息
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    # 获取相机内参
    intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
    intrinsics_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ])

    print("Camera Intrinsics Matrix:")
    print(intrinsics_matrix)

finally:
    # 停止管道
    pipeline.stop()


# 创建一个RealSense设备列表
'''
ctx = rs.context()
devices = ctx.query_devices()

# 遍历设备列表并打印每个设备的ID和类型
for i, dev in enumerate(devices):
    serial_number = dev.get_info(rs.camera_info.serial_number)
    camera_name = dev.get_info(rs.camera_info.name)
    print(f"RealSense相机{i+1}的ID为：{serial_number}")
    print(f"RealSense相机{i+1}的类型为：{camera_name}")

'''

