import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, SubsetRandomSampler
from model import RobotActionModel
from dataset import RobotActionDataset
import os
import argparse


def create_train_test_split(dataset, train_ratio=0.9, random_seed=42):
    """
    创建训练集和测试集的划分
    
    Args:
        dataset: 数据集
        train_ratio: 训练集比例，默认为0.9
        random_seed: 随机种子，用于可重复性
    
    Returns:
        train_sampler: 训练集采样器
        test_sampler: 测试集采样器
    """
    # 设置随机种子
    np.random.seed(random_seed)
    
    # 获取数据集大小
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    # 计算分割点
    split_idx = int(np.floor(train_ratio * dataset_size))
    
    # 打乱索引
    np.random.shuffle(indices)
    
    # 获取训练集和测试集的索引
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    
    # 创建采样器
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    return train_sampler, test_sampler


def train_model(model, train_loader, dataset, num_epochs=100, learning_rate=0.001):
    """
    训练模型的函数，包含训练和测试
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        dataset: 用于反归一化的原始数据集
        num_epochs: 训练的轮数
        learning_rate: 学习率
    
    Returns:
        model: 训练好的模型
        train_losses: 训练损失历史
        test_losses: 测试损失历史
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        
        for features, labels in train_loader:
            # 将数据移到设备上
            for key in features:
                features[key] = features[key].to(device)
            labels = labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(features)
            # 计算损失（物理量空间）
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # 计算训练平均损失
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # 打印训练信息
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        
    
    print("训练完成!")
    return model, train_losses


def evaluate_model(model, test_loader, dataset):
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
                features[key] = features[key].to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(features)
            
            # 转换为CPU上的numpy数组
            outputs = outputs.cpu().numpy()
            labels = labels.cpu().numpy()
            
            all_predictions.extend(outputs)
            all_ground_truth.extend(labels)
    
    # 转换为numpy数组
    predictions = np.array(all_predictions)
    ground_truth = np.array(all_ground_truth)
    
    # 计算每个输出维度的均方误差
    mse = np.mean((predictions - ground_truth) ** 2, axis=0)
    
    return predictions, ground_truth, mse


def plot_losses(train_losses, save_path='loss_curve.png'):
    """
    绘制训练和测试损失曲线
    
    Args:
        train_losses: 训练损失历史
        test_losses: 测试损失历史
        save_path: 保存图片的路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


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


# 主函数
if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='训练机器人动作模型')
    parser.add_argument('--position', type=int, default=1, choices=[1, 2, 3, 4], 
                       help='指定要训练的position (1,2,3,4)')
    parser.add_argument('--data_dir', type=str, default="./processed_data",
                       help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default="checkpoints",
                       help='模型和结果保存目录')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='训练批次大小')
    parser.add_argument('--num_epochs', type=int, default=2000,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 确定要训练的positions
    positions = [args.position] if args.position else [1, 2, 3, 4]
    
    for position in positions:
        print(f"\n=== 训练Position {position}的模型 ===")
        
        # 创建该position的保存目录
        position_dir = os.path.join(args.output_dir, f"position_{position}")
        os.makedirs(position_dir, exist_ok=True)
        
        # 加载特定position的数据
        dataset = RobotActionDataset(args.data_dir, normalize=True, target_position=position)
        
        if len(dataset) == 0:
            print(f"Position {position}没有有效数据，跳过训练")
            continue
        
        # 9:1划分训练集和测试集
        train_sampler, test_sampler = create_train_test_split(dataset, train_ratio=1.0)
        
        # 创建数据加载器
        train_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            sampler=train_sampler,
            drop_last=False
        )
        
        print(f"训练集大小: {len(train_sampler)}")
        
        # 创建模型
        model = RobotActionModel()
        
        # 训练模型
        model, train_losses = train_model(
            model, 
            train_loader, 
            dataset,
            num_epochs=args.num_epochs, 
            learning_rate=args.learning_rate
        )
        
        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_losses': train_losses,
        }, os.path.join(position_dir, "model.pth"))
        
        # 绘制损失曲线
        plot_losses(
            train_losses, 
            save_path=os.path.join(position_dir, "loss_curve.png")
        )

        # 评估模型（训练集）
        train_predictions, train_ground_truth, train_mse = evaluate_model(model, train_loader, dataset)
        print(f"\n训练集上的均方误差 (MSE):")
        print(f"  axis2_change: {train_mse[0]:.6f}")
        print(f"  axis4_change: {train_mse[1]:.6f}")
        print(f"  axis5_change: {train_mse[2]:.6f}")
        print(f"  平均: {np.mean(train_mse):.6f}")
        # 绘制预测值与真实值的对比（训练集）
        plot_predictions(
            train_predictions,
            train_ground_truth,
            save_path=os.path.join(position_dir, "train_predictions_vs_ground_truth.png")
        )
        # 保存评估结果（训练集）
        np.savez(
            os.path.join(position_dir, "train_evaluation_results.npz"),
            predictions=train_predictions,
            ground_truth=train_ground_truth,
            mse=train_mse
        )
        
        print(f"\nPosition {position}的模型和结果已保存到: {position_dir}")