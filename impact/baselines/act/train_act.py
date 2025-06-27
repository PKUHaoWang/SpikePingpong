import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy


import IPython
e = IPython.embed

from dataset import RobotActionDataset
from torch.utils.data import DataLoader


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_dataloaders(data_dir, position, batch_size):
    dataset = RobotActionDataset(data_dir, normalize=True, target_position=position)
    
    train_loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        drop_last=False
    )
    
    return train_loader

def evaluate_model(model, test_loader):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        dataset: 原始数据集，用于反归一化
    
    Returns:
        predictions: 所有测试样本的预测结果
        ground_truth: 所有测试样本的真实标签
        mse: 每个输出维度的均方误差
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_ground_truth = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            # 将数据移到设备上
            for key in features:
                features[key] = features[key].cuda()
            labels = labels.cuda()
            
            # 前向传播
            pred_action = model(features, None)
            
            # 转换为CPU上的numpy数组
            pred_action = pred_action.cpu().numpy()
            labels = labels.cpu().numpy()
            
            all_predictions.extend(pred_action)
            all_ground_truth.extend(labels)
    
    # 转换为numpy数组
    predictions = np.array(all_predictions)[:,0,:]
    ground_truth = np.array(all_ground_truth)
    # 计算每个输出维度的均方误差
    mse = np.mean((predictions - ground_truth) ** 2, axis=0)
    
    return predictions, ground_truth, mse


def forward_pass(data, policy):
    features, label = data
    for key in features:
        features[key] = features[key].cuda()
    label = label.cuda()
    return policy(features, label) # TODO remove None

def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(train_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)
        
        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    # 评估模型（训练集）
    
    train_predictions, train_ground_truth, train_mse = evaluate_model(policy, train_dataloader)
    print(f"\n训练集上的均方误差 (MSE):")
    print(f"  axis2_change: {train_mse[0]:.6f}")  # 第一个维度
    print(f"  axis4_change: {train_mse[1]:.6f}")  # 第二个维度
    print(f"  axis5_change: {train_mse[2]:.6f}")  # 第三个维度
    print(f"  平均: {np.mean(train_mse):.6f}")
    # 绘制预测值与真实值的对比（训练集）
    plot_predictions(
        train_predictions[:,:],
        train_ground_truth[:,:],
        save_path=os.path.join(ckpt_dir, "train_predictions_vs_ground_truth.png")
    )

    return best_ckpt_info
    
def plot_predictions(predictions, ground_truth, save_path='predictions_vs_ground_truth.png'):
    """
    绘制预测值与真实值的散点图
    
    Args:
        predictions: 预测值
        ground_truth: 真实值
        save_path: 保存图片的路径
    """
    output_names = ["axis2_change", "axis4_change", "axis5_change"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i in range(3):
        ax = axes[i]
        ax.scatter(ground_truth[:, i], predictions[:, i], alpha=0.5)
        ax.plot([ground_truth[:, i].min(), ground_truth[:, i].max()], 
                [ground_truth[:, i].min(), ground_truth[:, i].max()], 
                'r--')
        ax.set_xlabel('True Value')
        ax.set_ylabel('Predicted Value')
        ax.set_title(output_names[i])
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy

def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


def main(args):
    set_seed(1)
    
    data_dir = '../../processed_data'
    position = 1
    camera_names = range(1, 3)
    camera_names = [f'cam_{name:02d}' for name in camera_names]
    
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # fixed parameters
    state_dim = 3
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
    else:
        raise NotImplementedError
    
    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        # 'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        # 'real_robot': not is_sim
    }
        
    train_dataloader = get_dataloaders(data_dir, position, batch_size_train)
        
    best_ckpt_info = train_bc(train_dataloader, None, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    main(vars(parser.parse_args()))