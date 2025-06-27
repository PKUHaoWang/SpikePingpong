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


class RobotActionDataset(Dataset):
    """
    针对机器人动作数据的数据集类
    
    输入特征:
    - position: 前三帧的filtered_pos (9维)
    - velocity: 前三帧的filtered_vel (9维)
    - end_position: 预测的end_position (3维)
    - target_joint: 目标关节角度 (6维)
    
    输出标签:
    - axis2_change
    - axis4_change
    - axis5_change
    """
    
    def __init__(
        self, 
        data_dir: str,
        episode_pattern: str = "episode_*",
        transform=None,
        normalize: bool = True,
        target_position: Optional[int] = None
    ):
        """
        初始化数据集
        
        Args:
            data_dir: 包含所有episode的根目录
            episode_pattern: 用于匹配episode目录的模式
            transform: 可选的数据转换函数
            normalize: 是否对数据进行归一化
            target_position: 可选，指定要训练的目标位置（1,2,3,4），如果为None则使用所有数据
        """
        self.data_dir = data_dir
        self.episode_pattern = episode_pattern
        self.transform = transform
        self.normalize = normalize
        self.target_position = target_position
        
        if target_position is not None and target_position not in [1, 2, 3, 4]:
            raise ValueError("target_position必须是1,2,3,4中的一个，或者为None")
        
        # 查找所有符合条件的episode目录
        self.episode_paths = sorted(glob.glob(os.path.join(data_dir, episode_pattern)))[:245]
        print(f"找到 {len(self.episode_paths)} 个episode目录")
        
        if target_position is not None:
            print(f"将只加载end position为{target_position}的数据")
        
        # 预处理: 加载数据并进行特征提取
        self.positions = []  # 前三帧的位置
        self.velocities = []  # 前三帧的速度
        self.end_positions = []  # 预测的结束位置
        self.target_joints = []  # 目标关节角度
        self.landing_labels = []  # 球的最终落点标签 (1,2,3,4)
        self.labels = []  # 输出标签
        self.episode_ids = []
        
        # 记录无效数据
        self.invalid_episodes = []
        
        # 加载所有数据
        self._load_data()
        
        # 数据归一化
        if normalize and len(self.positions) > 0:
            self._normalize_data()
        
        # 转换为tensor
        self.positions = torch.tensor(np.array(self.positions), dtype=torch.float32)
        self.velocities = torch.tensor(np.array(self.velocities), dtype=torch.float32)
        self.end_positions = torch.tensor(np.array(self.end_positions), dtype=torch.float32)
        self.target_joints = torch.tensor(np.array(self.target_joints), dtype=torch.float32)
        self.landing_labels = torch.tensor(np.array(self.landing_labels), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)
        
        print(f"有效数据样本数: {len(self.positions)}")
        if target_position is not None:
            print(f"其中end position为{target_position}的样本数: {len(self.positions)}")
        print(f"位置维度: {self.positions.shape}")
        print(f"速度维度: {self.velocities.shape}")
        print(f"终点位置维度: {self.end_positions.shape}")
        print(f"目标关节角度维度: {self.target_joints.shape}")
        print(f"落点标签维度: {self.landing_labels.shape}")
        print(f"输出标签维度: {self.labels.shape}")
        
        # 保存数据统计信息
        # self._save_data_statistics()
    
    def _load_data(self):
        """加载所有episode的数据"""
        for ep_path in self.episode_paths:
            try:
                # 提取episode_id
                episode_id = int(os.path.basename(ep_path).split('_')[-1])
                
                # 检查数据有效性并加载
                data_tuple = self._load_episode(ep_path)
                if data_tuple is not None:
                    positions, velocities, end_position, target_joint, landing_label, joint_changes = data_tuple
                    self.positions.append(positions)
                    self.velocities.append(velocities)
                    self.end_positions.append(end_position)
                    self.target_joints.append(target_joint)
                    self.landing_labels.append(landing_label)
                    self.labels.append(joint_changes)
                    self.episode_ids.append(episode_id)
                else:
                    self.invalid_episodes.append(episode_id)
            except Exception as e:
                print(f"加载episode {ep_path} 失败: {str(e)}")
                self.invalid_episodes.append(os.path.basename(ep_path))
    
    def _load_episode(self, episode_path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, np.ndarray]]:
        """
        加载单个episode的数据
        
        Args:
            episode_path: episode目录路径
            
        Returns:
            positions: 3帧的位置数据
            velocities: 3帧的速度数据
            end_position: 预测的终点位置
            target_joint: 目标关节角度
            landing_label: 球的最终落点标签
            joint_changes: 输出标签 (axis2_change, axis4_change, axis5_change)
        """
        # 检查必要文件是否存在
        metadata_path = os.path.join(episode_path, "metadata.json")
        trajectory_path = os.path.join(episode_path, "l455_data", "trajectory.json")
        predictions_path = os.path.join(episode_path, "l455_data", "predictions.json")
        
        # 查找joints文件
        joints_files = sorted(glob.glob(os.path.join(episode_path, "l455_data", "robot", "joints_*.json")))
        if not joints_files:
            print(f"警告: {episode_path} 中没有找到joints文件")
            return None
        
        # 使用第一个joints文件
        joints_path = joints_files[0]
        
        # 检查文件是否存在
        if not all(os.path.exists(p) for p in [metadata_path, trajectory_path, predictions_path, joints_path]):
            missing = []
            if not os.path.exists(metadata_path): missing.append("metadata.json")
            if not os.path.exists(trajectory_path): missing.append("trajectory.json")
            if not os.path.exists(predictions_path): missing.append("predictions.json")
            if not os.path.exists(joints_path): missing.append("joints_*.json")
            print(f"警告: {episode_path} 缺少文件: {', '.join(missing)}")
            return None
        
        # 加载metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # 确保label不为0
        landing_label = metadata.get('label', 0)
        if landing_label == 0:
            print(f"警告: {episode_path} 的label为0，跳过")
            return None
            
        # 如果指定了target_position，检查是否匹配
        if self.target_position is not None and landing_label != self.target_position:
            return None
        
        # 加载trajectory数据
        with open(trajectory_path, 'r') as f:
            trajectory_data = json.load(f)
        
        # 确保至少有3帧数据
        frames = trajectory_data.get('frames', [])
        if len(frames) < 3:
            print(f"警告: {episode_path} 的trajectory帧数不足3帧，跳过")
            return None
        
        # 提取前3帧的filtered_pos和filtered_vel
        positions = []
        velocities = []
        for i in range(3):
            if i < len(frames):
                frame = frames[i]
                positions.extend(frame.get('filtered_pos', [0, 0, 0]))
                velocities.extend(frame.get('filtered_vel', [0, 0, 0]))
        
        # 将列表转换为numpy数组
        positions = np.array(positions, dtype=np.float32)
        velocities = np.array(velocities, dtype=np.float32)
        
        # 加载predictions数据
        with open(predictions_path, 'r') as f:
            predictions_data = json.load(f)
        
        # 提取end_position
        predictions = predictions_data.get('predictions', [])
        end_position = np.array([0, 0, 0], dtype=np.float32)
        if predictions:
            prediction = predictions[0].get('prediction', {})
            end_position = np.array(prediction.get('end_position', [0, 0, 0]), dtype=np.float32)
        
        # 加载joints数据
        with open(joints_path, 'r') as f:
            joints_data = json.load(f)
        
        # 提取hit_joint_changes
        hit_joint_changes = joints_data.get('hit_joint_changes', {})
        axis2_change = hit_joint_changes.get('axis2_change', 0)
        axis4_change = hit_joint_changes.get('axis4_change', 0)
        axis5_change = hit_joint_changes.get('axis5_change', 0)
        
        # 构建标签向量 [axis2_change, axis4_change, axis5_change]
        joint_changes = np.array([axis2_change, axis4_change, axis5_change], dtype=np.float32)
        
        # 提取target_joint
        target_joint = np.array(joints_data.get('target_joint', [0, 0, 0, 0, 0, 0]), dtype=np.float32)
        
        return positions, velocities, end_position, target_joint, landing_label, joint_changes
    
    def _normalize_data(self):
        """对数据进行最小-最大归一化处理（包括标签）"""
        # 计算各个特征组件的统计量
        positions_array = np.array(self.positions)
        velocities_array = np.array(self.velocities)
        end_positions_array = np.array(self.end_positions)
        target_joints_array = np.array(self.target_joints)
        landing_labels_array = np.array(self.landing_labels)
        labels_array = np.array(self.labels)
        
        # 计算并存储归一化参数
        self.positions_min = np.min(positions_array, axis=0)
        self.positions_max = np.max(positions_array, axis=0)
        self.positions_range = self.positions_max - self.positions_min
        self.positions_range[self.positions_range == 0] = 1.0  # 避免除零
        
        self.velocities_min = np.min(velocities_array, axis=0)
        self.velocities_max = np.max(velocities_array, axis=0)
        self.velocities_range = self.velocities_max - self.velocities_min
        self.velocities_range[self.velocities_range == 0] = 1.0  # 避免除零
        
        self.end_positions_min = np.min(end_positions_array, axis=0)
        self.end_positions_max = np.max(end_positions_array, axis=0)
        self.end_positions_range = self.end_positions_max - self.end_positions_min
        self.end_positions_range[self.end_positions_range == 0] = 1.0  # 避免除零
        
        self.target_joints_min = np.min(target_joints_array, axis=0)
        self.target_joints_max = np.max(target_joints_array, axis=0)
        self.target_joints_range = self.target_joints_max - self.target_joints_min
        self.target_joints_range[self.target_joints_range == 0] = 1.0  # 避免除零
        
        # 对于landing_labels，我们不归一化，因为它们是离散的类别标签
        
        self.label_min = np.min(labels_array, axis=0)
        self.label_max = np.max(labels_array, axis=0)
        self.label_range = self.label_max - self.label_min
        self.label_range[self.label_range == 0] = 1.0  # 避免除零
        #self.labels = (labels_array - self.label_min) / self.label_range
        
        # 归一化各个特征组件
        self.positions = (positions_array - self.positions_min) / self.positions_range
        self.velocities = (velocities_array - self.velocities_min) / self.velocities_range
        self.end_positions = (end_positions_array - self.end_positions_min) / self.end_positions_range
        self.target_joints = (target_joints_array - self.target_joints_min) / self.target_joints_range
        # landing_labels保持不变
        # 保存归一化参数，用于后续转换
        self.normalization_params = {
            'positions_min': self.positions_min,
            'positions_max': self.positions_max,
            'velocities_min': self.velocities_min,
            'velocities_max': self.velocities_max,
            'end_positions_min': self.end_positions_min,
            'end_positions_max': self.end_positions_max,
            'target_joints_min': self.target_joints_min,
            'target_joints_max': self.target_joints_max,
            'label_min': self.label_min,
            'label_max': self.label_max
        }
    
    def denormalize_labels(self, normalized_labels: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """将标准化的标签转换回原始范围"""
        if not self.normalize:
            return normalized_labels
        
        if isinstance(normalized_labels, torch.Tensor):
            normalized_labels = normalized_labels.detach().cpu().numpy()
        
        return normalized_labels * self.label_range + self.label_min
    
    def save_normalization_params(self, save_path: str):
        """保存归一化参数"""
        if not self.normalize:
            print("警告: 数据未归一化，无需保存参数")
            return
        
        np.savez(
            save_path,
            positions_min=self.positions_min,
            positions_max=self.positions_max,
            velocities_min=self.velocities_min,
            velocities_max=self.velocities_max,
            end_positions_min=self.end_positions_min,
            end_positions_max=self.end_positions_max,
            target_joints_min=self.target_joints_min,
            target_joints_max=self.target_joints_max,
            label_min=self.label_min,
            label_max=self.label_max
        )
        print(f"归一化参数已保存到 {save_path}")
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        position = self.positions[idx]
        velocity = self.velocities[idx]
        end_position = self.end_positions[idx]
        target_joint = self.target_joints[idx]
        landing_label = self.landing_labels[idx]
        label = self.labels[idx]
        
        # 返回所有特征组件和标签
        features = {
            'position': position,
            'velocity': velocity,
            'end_position': end_position,
            'target_joint': target_joint,
            'landing_label': landing_label
        }
        
        if self.transform:
            features = self.transform(features)
        
        return features, label
    
    def _save_data_statistics(self):
        """保存数据统计信息"""
        # 这里可以根据需要实现保存数据统计信息的逻辑
        print("数据统计信息已保存")

    
    
    
if __name__ == "__main__":
    # 示例用法
    data_dir = "./data"
    
    # 测试每个position的数据
    for position in [1, 2, 3, 4]:
        print(f"\n=== 测试加载Position {position}的数据 ===")
        dataset = RobotActionDataset(data_dir, normalize=True, target_position=position)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # 获取一个样本进行测试
        if len(dataset) > 0:
            features, labels = next(iter(dataloader))
            print(f"\nPosition {position} 数据样本示例:")
            print(f"Position shape: {features['position'].shape}")
            print(f"Velocity shape: {features['velocity'].shape}")
            print(f"End position shape: {features['end_position'].shape}")
            print(f"Target joint shape: {features['target_joint'].shape}")
            print(f"Spike image shape: {features['spike_image'].shape}")
            print(f"Landing label: {features['landing_label'].tolist()}")
            print(f"Labels shape: {labels.shape}")
            print(f"Labels value: {labels.numpy().flatten()}")
            
            # # 收集所有labels
            # all_labels = []
            # for batch_features, batch_labels in dataloader:
            #     all_labels.append(batch_labels.numpy())
            
            # # 合并所有labels
            # all_labels = np.vstack(all_labels)
            # print(f"\nPosition {position} 总数据量: {all_labels.shape[0]}")
            
            # # 保存为npy文件
            # np.save(f'./position_{position}_labels.npy', all_labels)
            # print(f"已保存labels")
        else:
            print(f"\nPosition {position} 没有找到有效数据")
    
    # 输出数据集的基本统计信息
    # 测试加载所有数据
    print("\n=== 测试加载所有数据 ===")
    dataset_all = RobotActionDataset(data_dir, normalize=True)
    
    # 创建数据加载器
    dataloader_all = DataLoader(dataset_all, batch_size=1, shuffle=True)
    print("\n=== 数据集统计信息 ===")
    print(f"总样本数: {len(dataset_all)}")
    for position in [1, 2, 3, 4]:
        dataset = RobotActionDataset(data_dir, normalize=True, target_position=position)
        print(f"Position {position} 样本数: {len(dataset)}")