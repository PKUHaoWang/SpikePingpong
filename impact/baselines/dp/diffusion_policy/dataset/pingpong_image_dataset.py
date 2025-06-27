from typing import Dict
import torch
import numpy as np
import copy
import os
import glob
import json
from PIL import Image
import cv2
from torchvision import transforms
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class PingPongImageDataset(BaseImageDataset):
    def __init__(self,
            data_dir,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            episode_pattern="episode_*",
            normalize=True,
            image_size=(96, 96),
            use_preprocess=False,
            target_position=None
            ):
        super().__init__()
        self.data_dir = data_dir
        self.episode_pattern = episode_pattern
        self.normalize = normalize
        self.target_position = target_position
        self.image_size = image_size
        self.use_preprocess = use_preprocess
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes

        self.img_transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])

        # 查找所有 episode
        self.episode_paths = sorted(glob.glob(os.path.join(data_dir, episode_pattern)))
        self.samples = []
        for ep_path in self.episode_paths:
            spike_dir = os.path.join(ep_path, "spike_images")
            images = sorted(glob.glob(os.path.join(spike_dir, "*.png")))
            if len(images) < horizon:
                continue
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
        # 划分训练/验证
        np.random.seed(seed)
        indices = np.arange(len(self.samples))
        np.random.shuffle(indices)
        n_val = int(len(indices) * val_ratio)
        self.val_indices = indices[:n_val]
        self.train_indices = indices[n_val:]
        if max_train_episodes is not None:
            self.train_indices = self.train_indices[:max_train_episodes]
        self.is_val = False
        # 归一化参数
        self.normalization_params = None
        if normalize and len(self.samples) > 0:
            self.normalization_params = self._compute_normalization_params()

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.is_val = True
        return val_set

    def _compute_normalization_params(self):
        pos_list, vel_list, end_pos_list, target_joint_list, label_list = [], [], [], [], []
        for idx in (self.val_indices if self.is_val else self.train_indices):
            ep_path = self.samples[idx]
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

    def get_normalizer(self, mode='limits', **kwargs):
        # 这里只对低维特征做normalizer，图片用[-1,1]归一化
        data = {}
        if self.normalization_params is not None:
            np_params = self.normalization_params
            data['position'] = np.stack([
                self._normalize(np.array([np_params['positions_min']]), np_params['positions_min'], np_params['positions_max']),
                self._normalize(np.array([np_params['positions_max']]), np_params['positions_min'], np_params['positions_max'])
            ])
            data['velocity'] = np.stack([
                self._normalize(np.array([np_params['velocities_min']]), np_params['velocities_min'], np_params['velocities_max']),
                self._normalize(np.array([np_params['velocities_max']]), np_params['velocities_min'], np_params['velocities_max'])
            ])
            data['end_position'] = np.stack([
                self._normalize(np.array([np_params['end_positions_min']]), np_params['end_positions_min'], np_params['end_positions_max']),
                self._normalize(np.array([np_params['end_positions_max']]), np_params['end_positions_min'], np_params['end_positions_max'])
            ])
            data['target_joint'] = np.stack([
                self._normalize(np.array([np_params['target_joints_min']]), np_params['target_joints_min'], np_params['target_joints_max']),
                self._normalize(np.array([np_params['target_joints_max']]), np_params['target_joints_min'], np_params['target_joints_max'])
            ])
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def _normalize(self, arr, arr_min, arr_max):
        arr_range = arr_max - arr_min
        arr_range[arr_range == 0] = 1.0
        return (arr - arr_min) / arr_range

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

    def __len__(self):
        indices = self.val_indices if self.is_val else self.train_indices
        return len(indices)

    def _sample_to_data(self, ep_path):
        data_tuple = self._load_episode(ep_path)
        if data_tuple is None:
            raise RuntimeError(f"加载episode失败: {ep_path}")
        positions, velocities, end_position, target_joint, landing_label, joint_changes = data_tuple
        if self.normalize and self.normalization_params is not None:
            np_params = self.normalization_params
            positions = self._normalize(positions, np_params['positions_min'], np_params['positions_max'])
            velocities = self._normalize(velocities, np_params['velocities_min'], np_params['velocities_max'])
            end_position = self._normalize(end_position, np_params['end_positions_min'], np_params['end_positions_max'])
            target_joint = self._normalize(target_joint, np_params['target_joints_min'], np_params['target_joints_max'])
        spike_dir = os.path.join(ep_path, "spike_images")
        image_paths = sorted(glob.glob(os.path.join(spike_dir, "*.png")))
        image_paths = image_paths[:self.horizon]
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
        spike_imgs = torch.stack(spike_imgs, dim=0)  # (T, C, H, W)
        data = {
            'obs': {
                'image': spike_imgs,  # T, 3, H, W
                'position': torch.tensor(positions, dtype=torch.float32),
                'velocity': torch.tensor(velocities, dtype=torch.float32),
                'end_position': torch.tensor(end_position, dtype=torch.float32),
                'target_joint': torch.tensor(target_joint, dtype=torch.float32),
            },
            'action': torch.tensor(joint_changes, dtype=torch.float32)  # T, Da (这里只是示例)
        }
        return data

    @staticmethod
    def preprocess_image(img, h=10, h_color=10, template_window=7, search_window=21, alpha=1.0, beta=15):
        img = cv2.fastNlMeansDenoisingColored(
            img, None, h, h_color, template_window, search_window
        )
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(pil_img)
        return img_tensor

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        indices = self.val_indices if self.is_val else self.train_indices
        ep_path = self.samples[indices[idx]]
        data = self._sample_to_data(ep_path)
        return dict_apply(data, lambda x: x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32))


def test():
    data_dir = os.path.expanduser('~/dev/pingpong_data')
    dataset = PingPongImageDataset(data_dir, horizon=8)
    print('样本数:', len(dataset))
    sample = dataset[0]
    print('sample keys:', sample.keys())
    print('obs keys:', sample['obs'].keys())
    print('image shape:', sample['obs']['image'].shape) 