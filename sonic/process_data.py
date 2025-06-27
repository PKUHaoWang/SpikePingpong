#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import glob
from get_delta_pos import segment_colors

def process_raw_data():
    # 创建输出目录
    output_dir = "processed_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有原始数据文件夹
    raw_data_dirs = glob.glob("rawdata/*/")

    episode_id = 0
    
    for episode_idx, data_dir in enumerate(sorted(raw_data_dirs)):
        # 读取trajectory.json
        trajectory_path = os.path.join(data_dir, "trajectory.json")
        if not os.path.exists(trajectory_path):
            print(f"跳过 {data_dir}: 未找到trajectory.json")
            continue
            
        with open(trajectory_path, 'r') as f:
            trajectory_data = json.load(f)
            
        # 获取前三帧数据
        first_three_frames = trajectory_data['frames'][:3]
        
        # 读取predictions.json
        predictions_path = os.path.join(data_dir, "predictions.json")
        if not os.path.exists(predictions_path):
            print(f"跳过 {data_dir}: 未找到predictions.json")
            continue
            
        with open(predictions_path, 'r') as f:
            predictions_data = json.load(f)
            
        # 获取end position
        end_position = None
        if predictions_data.get('predictions') and len(predictions_data['predictions']) > 0:
            end_position = predictions_data['predictions'][0]['prediction'].get('end_position')
        
        # 获取图像文件
        image_files = glob.glob(os.path.join(data_dir, "*.png"))
        if not image_files:
            print(f"跳过 {data_dir}: 未找到图像文件")
            continue
            
        # 使用get_delta_pos.py处理图像
        image_path = sorted(image_files)[0]  # 使用第一张图片
        
        # 获取数据目录名作为输出前缀
        output_prefix = os.path.basename(os.path.dirname(data_dir))
        
        # 创建临时输出目录
        temp_output_dir = os.path.join(output_dir, "visualizations")
        if not os.path.exists(temp_output_dir):
            os.makedirs(temp_output_dir)
        
        _, ball_positions = segment_colors(
            image_path,
            output_dir=temp_output_dir,
            output_prefix=output_prefix,
            color1=(65, 31, 31),
            color2=(79, 58, 34),
            tolerance1=25,
            tolerance2=12,
            min_area=100,
            center_ratio=0.4,
            mm_per_pixel=2.2828
        )
        
        # 如果球位置为空，跳过这条数据
        if not ball_positions:
            print(f"跳过 {data_dir}: 未检测到球的位置")
            continue
        
        # 创建新的数据格式
        processed_data = {
            "episode_id": episode_idx,
            "first_three_frames": first_three_frames,
            "end_position": end_position,
            "ball_positions": ball_positions
        }
        
        # 保存处理后的数据
        output_file = os.path.join(output_dir, f"episode_{episode_id}.json")
        episode_id += 1
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
            
        print(f"处理完成: {output_file}")

if __name__ == "__main__":
    process_raw_data()
