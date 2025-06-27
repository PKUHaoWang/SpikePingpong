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


class SpikeDataset(Dataset):
    """
    针对带 spike image 的机器人动作数据集，仿照 RobotActionSpikeDataset 实现
    """
    def __init__(self, data_dir, episode_pattern="episode_*", normalize=True, target_position=None, image_size=(224, 224), use_preprocess=False):
        self.data_dir = data_dir
        self.episode_pattern = episode_pattern
        self.normalize = normalize
        self.target_position = target_position
        self.image_size = image_size
        self.use_preprocess = use_preprocess

        # 图像预处理
        self.img_transform = transforms.Compose([
             transforms.Resize(self.image_size),
             transforms.ToTensor(),  # (C, H, W), [0,1]
         ])

        # 查找所有 episode
        self.episode_paths = sorted(glob.glob(os.path.join(data_dir, episode_pattern)))[:245]
        self.samples = []
        for ep_path in self.episode_paths:
            # 检查 spike_images 是否存在且有3张
            spike_dir = os.path.join(ep_path, "spike_images")
            images = sorted(glob.glob(os.path.join(spike_dir, "*.png")))
            if len(images) < 3:
                continue
            images = images[:3]
            # 检查label
            metadata_path = os.path.join(ep_path, "metadata.json")
            if not os.path.exists(metadata_path):
                continue
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            label = metadata.get("label", 0)
            if label == 0:
                continue
            if target_position is not None and label != target_position:
                continue
            self.samples.append(ep_path)

        # 归一化参数统计（可选）
        self.normalization_params = None
        if normalize and len(self.samples) > 0:
            self.normalization_params = self._compute_normalization_params()

    def _load_episode(self, episode_path: str):
        metadata_path = os.path.join(episode_path, "metadata.json")
        trajectory_path = os.path.join(episode_path, "l455_data", "trajectory.json")
        predictions_path = os.path.join(episode_path, "l455_data", "predictions.json")
        joints_files = sorted(glob.glob(os.path.join(episode_path, "l455_data", "robot", "joints_*.json")))
        if not joints_files:
            return None
        joints_path = joints_files[0]
        if not all(os.path.exists(p) for p in [metadata_path, trajectory_path, predictions_path, joints_path]):
            return None
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        landing_label = metadata.get('label', 0)
        if landing_label == 0:
            return None
        if self.target_position is not None and landing_label != self.target_position:
            return None
        with open(trajectory_path, 'r') as f:
            trajectory_data = json.load(f)
        frames = trajectory_data.get('frames', [])
        if len(frames) < 3:
            return None
        positions = []
        velocities = []
        for i in range(3):
            if i < len(frames):
                frame = frames[i]
                filtered_pos = frame.get('filtered_pos')
                if filtered_pos is None:
                    filtered_pos = [0, 0, 0]
                positions.extend(filtered_pos)
                filtered_vel = frame.get('filtered_vel')
                if filtered_vel is None:
                    filtered_vel = [0, 0, 0]
                velocities.extend(filtered_vel)
        positions = np.array(positions, dtype=np.float32)
        velocities = np.array(velocities, dtype=np.float32)
        with open(predictions_path, 'r') as f:
            predictions_data = json.load(f)
        predictions = predictions_data.get('predictions', [])
        end_position = np.array([0, 0, 0], dtype=np.float32)
        if predictions:
            prediction = predictions[0].get('prediction', {})
            end_position = np.array(prediction.get('end_position', [0, 0, 0]), dtype=np.float32)
        with open(joints_path, 'r') as f:
            joints_data = json.load(f)
        hit_joint_changes = joints_data.get('hit_joint_changes', {})
        axis2_change = hit_joint_changes.get('axis2_change', 0)
        axis4_change = hit_joint_changes.get('axis4_change', 0)
        axis5_change = hit_joint_changes.get('axis5_change', 0)
        joint_changes = np.array([axis2_change, axis4_change, axis5_change], dtype=np.float32)
        target_joint = np.array(joints_data.get('target_joint', [0, 0, 0, 0, 0, 0]), dtype=np.float32)
        return positions, velocities, end_position, target_joint, landing_label, joint_changes

    def _compute_normalization_params(self):
        # 只遍历一次所有样本，统计最大最小值
        pos_list, vel_list, end_pos_list, target_joint_list, label_list = [], [], [], [], []
        for ep_path in self.samples:
            data_tuple = self._load_episode(ep_path)
            if data_tuple is not None:
                positions, velocities, end_position, target_joint, landing_label, joint_changes = data_tuple
                pos_list.append(positions)
                vel_list.append(velocities)
                end_pos_list.append(end_position)
                target_joint_list.append(target_joint)
                label_list.append(joint_changes)
        if len(pos_list) == 0:
            return None
        pos_arr = np.array(pos_list)
        vel_arr = np.array(vel_list)
        end_pos_arr = np.array(end_pos_list)
        target_joint_arr = np.array(target_joint_list)
        label_arr = np.array(label_list)
        return {
            'positions_min': np.min(pos_arr, axis=0),
            'positions_max': np.max(pos_arr, axis=0),
            'velocities_min': np.min(vel_arr, axis=0),
            'velocities_max': np.max(vel_arr, axis=0),
            'end_positions_min': np.min(end_pos_arr, axis=0),
            'end_positions_max': np.max(end_pos_arr, axis=0),
            'target_joints_min': np.min(target_joint_arr, axis=0),
            'target_joints_max': np.max(target_joint_arr, axis=0),
            'label_min': np.min(label_arr, axis=0),
            'label_max': np.max(label_arr, axis=0)
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
        # 动态加载特征
        data_tuple = self._load_episode(ep_path)
        if data_tuple is None:
            raise RuntimeError(f"加载episode失败: {ep_path}")
        positions, velocities, end_position, target_joint, landing_label, joint_changes = data_tuple
        # 归一化
        if self.normalize and self.normalization_params is not None:
            np_params = self.normalization_params
            positions = self._normalize(positions, np_params['positions_min'], np_params['positions_max'])
            velocities = self._normalize(velocities, np_params['velocities_min'], np_params['velocities_max'])
            end_position = self._normalize(end_position, np_params['end_positions_min'], np_params['end_positions_max'])
            target_joint = self._normalize(target_joint, np_params['target_joints_min'], np_params['target_joints_max'])
            # joint_changes/label 归一化可选
        # 读取图片
        spike_dir = os.path.join(ep_path, "spike_images")
        image_paths = sorted(glob.glob(os.path.join(spike_dir, "*.png")))
        image_paths = image_paths[:3]
        spike_imgs = []
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if self.use_preprocess:
                img_tensor = self.preprocess_image(img)
            else:
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img_tensor = self.img_transform(pil_img)
            spike_imgs.append(img_tensor)
        spike_imgs = torch.stack(spike_imgs, dim=0)
        features = {
            'position': torch.tensor(positions, dtype=torch.float32),
            'velocity': torch.tensor(velocities, dtype=torch.float32),
            'end_position': torch.tensor(end_position, dtype=torch.float32),
            'target_joint': torch.tensor(target_joint, dtype=torch.float32),
            'spike_image': spike_imgs,
            'landing_label': torch.tensor(landing_label, dtype=torch.float32)
        }
        label = torch.tensor(joint_changes, dtype=torch.float32)
        return features, label
    
    
class RobotActionDataset(Dataset):
    """
    针对带 spike image 的机器人动作数据集，仿照 RobotActionSpikeDataset 实现
    """
    def __init__(self, data_dir, episode_pattern="episode_*", normalize=True, target_position=None, image_size=(224, 224), use_preprocess=False):
        self.data_dir = data_dir
        self.episode_pattern = episode_pattern
        self.normalize = normalize
        self.target_position = target_position
        self.image_size = image_size
        self.use_preprocess = use_preprocess

        # 图像预处理
        self.img_transform = transforms.Compose([
             transforms.Resize(self.image_size),
             transforms.ToTensor(),  # (C, H, W), [0,1]
         ])

        # 查找所有 episode
        self.episode_paths = sorted(glob.glob(os.path.join(data_dir, episode_pattern)))[:245]
        self.samples = []
        for ep_path in self.episode_paths:
            # 检查 l455 图像是否存在且有3张
            l455_dir = os.path.join(ep_path, "l455_data", "l455")
            images = sorted(glob.glob(os.path.join(l455_dir, "frame_*.png")))
            if len(images) < 3:
                continue
            images = images[:3]
            # 检查label
            metadata_path = os.path.join(ep_path, "metadata.json")
            if not os.path.exists(metadata_path):
                continue
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            label = metadata.get("label", 0)
            if label == 0:
                continue
            if target_position is not None and label != target_position:
                continue
            self.samples.append(ep_path)

        # 归一化参数统计（可选）
        self.normalization_params = None
        if normalize and len(self.samples) > 0:
            self.normalization_params = self._compute_normalization_params()

    def _load_episode(self, episode_path: str):
        metadata_path = os.path.join(episode_path, "metadata.json")
        trajectory_path = os.path.join(episode_path, "l455_data", "trajectory.json")
        predictions_path = os.path.join(episode_path, "l455_data", "predictions.json")
        joints_files = sorted(glob.glob(os.path.join(episode_path, "l455_data", "robot", "joints_*.json")))
        if not joints_files:
            return None
        joints_path = joints_files[0]
        if not all(os.path.exists(p) for p in [metadata_path, trajectory_path, predictions_path, joints_path]):
            return None
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        landing_label = metadata.get('label', 0)
        if landing_label == 0:
            return None
        if self.target_position is not None and landing_label != self.target_position:
            return None
        with open(trajectory_path, 'r') as f:
            trajectory_data = json.load(f)
        frames = trajectory_data.get('frames', [])
        if len(frames) < 3:
            return None
        positions = []
        velocities = []
        for i in range(3):
            if i < len(frames):
                frame = frames[i]
                filtered_pos = frame.get('filtered_pos')
                if filtered_pos is None:
                    filtered_pos = [0, 0, 0]
                positions.extend(filtered_pos)
                filtered_vel = frame.get('filtered_vel')
                if filtered_vel is None:
                    filtered_vel = [0, 0, 0]
                velocities.extend(filtered_vel)
        positions = np.array(positions, dtype=np.float32)
        velocities = np.array(velocities, dtype=np.float32)
        with open(predictions_path, 'r') as f:
            predictions_data = json.load(f)
        predictions = predictions_data.get('predictions', [])
        end_position = np.array([0, 0, 0], dtype=np.float32)
        if predictions:
            prediction = predictions[0].get('prediction', {})
            end_position = np.array(prediction.get('end_position', [0, 0, 0]), dtype=np.float32)
        with open(joints_path, 'r') as f:
            joints_data = json.load(f)
        hit_joint_changes = joints_data.get('hit_joint_changes', {})
        axis2_change = hit_joint_changes.get('axis2_change', 0)
        axis4_change = hit_joint_changes.get('axis4_change', 0)
        axis5_change = hit_joint_changes.get('axis5_change', 0)
        joint_changes = np.array([axis2_change, axis4_change, axis5_change], dtype=np.float32)
        target_joint = np.array(joints_data.get('target_joint', [0, 0, 0, 0, 0, 0]), dtype=np.float32)
        return positions, velocities, end_position, target_joint, landing_label, joint_changes

    def _compute_normalization_params(self):
        # 只遍历一次所有样本，统计最大最小值
        pos_list, vel_list, end_pos_list, target_joint_list, label_list = [], [], [], [], []
        for ep_path in self.samples:
            data_tuple = self._load_episode(ep_path)
            if data_tuple is not None:
                positions, velocities, end_position, target_joint, landing_label, joint_changes = data_tuple
                pos_list.append(positions)
                vel_list.append(velocities)
                end_pos_list.append(end_position)
                target_joint_list.append(target_joint)
                label_list.append(joint_changes)
        if len(pos_list) == 0:
            return None
        pos_arr = np.array(pos_list)
        vel_arr = np.array(vel_list)
        end_pos_arr = np.array(end_pos_list)
        target_joint_arr = np.array(target_joint_list)
        label_arr = np.array(label_list)
        return {
            'positions_min': np.min(pos_arr, axis=0),
            'positions_max': np.max(pos_arr, axis=0),
            'velocities_min': np.min(vel_arr, axis=0),
            'velocities_max': np.max(vel_arr, axis=0),
            'end_positions_min': np.min(end_pos_arr, axis=0),
            'end_positions_max': np.max(end_pos_arr, axis=0),
            'target_joints_min': np.min(target_joint_arr, axis=0),
            'target_joints_max': np.max(target_joint_arr, axis=0),
            'label_min': np.min(label_arr, axis=0),
            'label_max': np.max(label_arr, axis=0)
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
        # 动态加载特征
        data_tuple = self._load_episode(ep_path)
        if data_tuple is None:
            raise RuntimeError(f"加载episode失败: {ep_path}")
        positions, velocities, end_position, target_joint, landing_label, joint_changes = data_tuple
        # 归一化
        if self.normalize and self.normalization_params is not None:
            np_params = self.normalization_params
            positions = self._normalize(positions, np_params['positions_min'], np_params['positions_max'])
            velocities = self._normalize(velocities, np_params['velocities_min'], np_params['velocities_max'])
            end_position = self._normalize(end_position, np_params['end_positions_min'], np_params['end_positions_max'])
            target_joint = self._normalize(target_joint, np_params['target_joints_min'], np_params['target_joints_max'])
            # joint_changes/label 归一化可选
        # 读取图片
        l455_dir = os.path.join(ep_path, "l455_data", "l455")
        image_paths = sorted(glob.glob(os.path.join(l455_dir, "frame_*.png")))
        image_paths = image_paths[:3]
        l455_imgs = []
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            img = np.array(img)
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if self.use_preprocess:
                img_tensor = self.preprocess_image(img)
            else:
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                img_tensor = self.img_transform(pil_img)
            l455_imgs.append(img_tensor)
        l455_imgs = torch.stack(l455_imgs, dim=0)
        features = {
            'position': torch.tensor(positions, dtype=torch.float32),
            'velocity': torch.tensor(velocities, dtype=torch.float32),
            'end_position': torch.tensor(end_position, dtype=torch.float32),
            'target_joint': torch.tensor(target_joint, dtype=torch.float32),
            'l455_image': l455_imgs,
            'landing_label': torch.tensor(landing_label, dtype=torch.float32)
        }
        label = torch.tensor(joint_changes, dtype=torch.float32)
        return features, label

if __name__ == "__main__":
    # 测试数据集
    data_dir = "../../processed_data"  # 请根据实际数据路径修改
    dataset = RobotActionDataset(
        data_dir=data_dir,
        normalize=True,
        use_preprocess=True,
        target_position=1
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 测试获取一个样本
    features, label = dataset[0]
    print("\n特征信息:")
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"{key}: {value}")
    
    print(f"\n标签信息: shape={label.shape}, dtype={label.dtype}")
    print(f"标签值: {label}")
    
    # 测试数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2
    )
    
    print("\n测试数据加载器:")
    for batch_idx, (batch_features, batch_labels) in enumerate(dataloader):
        print(f"\n批次 {batch_idx}:")
        for key, value in batch_features.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
        print(f"labels: shape={batch_labels.shape}, dtype={batch_labels.dtype}")
        
        if batch_idx >= 1:  # 只测试前两个批次
            break