# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 21:38:48 2024

@author: Houch
"""

import pyrealsense2 as rs
import numpy as np
import cv2

# 配置管道
pipeline = rs.pipeline()
config = rs.config()

# 配置流
config.enable_stream(rs.stream.color,960, 540, rs.format.bgr8, 60)

# 启动管道
pipeline.start(config)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码
out = cv2.VideoWriter('output.mp4', fourcc, 60.0, (960, 540))

try:
    while True:
        # 等待帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # 将帧转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())

        # 写入视频文件
        out.write(color_image)

        # 显示图像
        cv2.imshow('Color Image', color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # 停止管道
    pipeline.stop()
    out.release()
    cv2.destroyAllWindows()
