import os
import json
import numpy as np
import torch
import cv2 
import glob
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image
from torchvision import transforms

class SpikeRefineDataset(Dataset):
    """
    针对处理后的机器人动作数据集
    """
    def __init__(self, data_dir, episode_pattern="episode_*.json", normalize=True, image_size=(224, 224)):
        self.data_dir = data_dir
        self.episode_pattern = episode_pattern
        self.normalize = normalize
        self.image_size = image_size

        # 图像预处理
        self.img_transform = transforms.Compose([
             transforms.Resize(self.image_size),
             transforms.ToTensor(),  # (C, H, W), [0,1]
         ])

        # 查找所有 episode
        self.episode_paths = sorted(glob.glob(os.path.join(data_dir, episode_pattern)))
        self.samples = []
        for ep_path in self.episode_paths:
            with open(ep_path, 'r') as f:
                data = json.load(f)
                if data.get('first_three_frames') and data.get('end_position') and data.get('ball_positions'):
                    self.samples.append(ep_path)

        # 归一化参数统计（可选）
        self.normalization_params = None
        if normalize and len(self.samples) > 0:
            self.normalization_params = self._compute_normalization_params()

    def _load_episode(self, episode_path: str):
        with open(episode_path, 'r') as f:
            data = json.load(f)
            
        # 获取前三帧数据
        frames = data.get('first_three_frames', [])
        if len(frames) < 3:
            return None
            
        # 提取位置和速度信息
        positions = []
        velocities = []
        for frame in frames:
            filtered_pos = frame.get('filtered_pos', [0, 0, 0])
            filtered_vel = frame.get('filtered_vel', [0, 0, 0])
            positions.extend(filtered_pos)
            velocities.extend(filtered_vel)
            
        positions = np.array(positions, dtype=np.float32)
        velocities = np.array(velocities, dtype=np.float32)
        
        # 获取end position
        end_position = np.array(data.get('end_position', [0, 0, 0]), dtype=np.float32)
        
        # 获取ball positions
        ball_positions = data.get('ball_positions', [])
        if not ball_positions:
            return None
            
        # 将ball positions转换为numpy数组，确保形状正确
        ball_positions = np.array(ball_positions, dtype=np.float32)
        if len(ball_positions.shape) == 1:
            ball_positions = ball_positions.reshape(-1, 2)  # 确保是 (N, 2) 形状
        
        # 只取第一个球的位置
        ball_position = ball_positions[0] if len(ball_positions) > 0 else np.array([0, 0], dtype=np.float32)
        
        return positions, velocities, end_position, ball_position

    def _compute_normalization_params(self):
        # 只遍历一次所有样本，统计最大最小值
        pos_list, vel_list, end_pos_list = [], [], []
        for ep_path in self.samples:
            data_tuple = self._load_episode(ep_path)
            if data_tuple is not None:
                positions, velocities, end_position, _ = data_tuple
                pos_list.append(positions)
                vel_list.append(velocities)
                end_pos_list.append(end_position)
                
        if len(pos_list) == 0:
            return None
            
        pos_arr = np.array(pos_list)
        vel_arr = np.array(vel_list)
        end_pos_arr = np.array(end_pos_list)
        
        return {
            'positions_min': np.min(pos_arr, axis=0),
            'positions_max': np.max(pos_arr, axis=0),
            'velocities_min': np.min(vel_arr, axis=0),
            'velocities_max': np.max(vel_arr, axis=0),
            'end_positions_min': np.min(end_pos_arr, axis=0),
            'end_positions_max': np.max(end_pos_arr, axis=0)
        }

    def _normalize(self, arr, arr_min, arr_max):
        arr_range = arr_max - arr_min
        arr_range[arr_range == 0] = 1.0
        return (arr - arr_min) / arr_range

    @staticmethod
    def preprocess_image(img, h=10, h_color=10, template_window=7, search_window=21, alpha=1.0, beta=15):
        # 非局部均值去噪
        img = cv2.fastNlMeansDenoisingColored(
            img, None, h, h_color, template_window, search_window
        )
        # 对比度和亮度调整
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
        pil_img = Image.fromarray(img)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(pil_img)
        return img_tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_path = self.samples[idx]
        # 加载特征
        data_tuple = self._load_episode(ep_path)
        if data_tuple is None:
            raise RuntimeError(f"加载episode失败: {ep_path}")
            
        positions, velocities, end_position, ball_position = data_tuple
        
        # 归一化特征
        if self.normalize and self.normalization_params is not None:
            np_params = self.normalization_params
            positions = self._normalize(positions, np_params['positions_min'], np_params['positions_max'])
            velocities = self._normalize(velocities, np_params['velocities_min'], np_params['velocities_max'])
            end_position = self._normalize(end_position, np_params['end_positions_min'], np_params['end_positions_max'])
            
        features = {
            'position': torch.tensor(positions, dtype=torch.float32),
            'velocity': torch.tensor(velocities, dtype=torch.float32),
            'end_position': torch.tensor(end_position, dtype=torch.float32)
        }
        
        label = torch.tensor(ball_position, dtype=torch.float32)
        return features, label

if __name__ == "__main__":
    # 创建数据集实例
    dataset = RobotActionDataset(
        data_dir="processed_data",
        normalize=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0
    )
    
    # 测试数据加载
    for batch_idx, (features, labels) in enumerate(dataloader):
        print(f"\n批次 {batch_idx + 1}:")
        print("特征:")
        print(f"- 位置形状: {features['position'].shape}")
        print(f"- 速度形状: {features['velocity'].shape}")
        print(f"- 终点位置形状: {features['end_position'].shape}")
        print(f"标签形状: {labels.shape}")
        print(f"标签内容: {labels}")
        
        # 只打印前两个批次
        if batch_idx >= 1:
            break
            
    # 打印归一化参数
    if dataset.normalization_params:
        print("\n归一化参数:")
        for key, value in dataset.normalization_params.items():
            print(f"{key}: {value}")