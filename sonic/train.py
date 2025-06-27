import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import SpikeRefineModel
from dataset import SpikeRefineDataset
import os
import argparse


def train_model(model, train_loader, num_epochs=100, learning_rate=0.001):
    """
    训练模型的函数
    
    Args:
        model: 要训练的模型
        train_loader: 训练数据加载器
        num_epochs: 训练的轮数
        learning_rate: 学习率
    
    Returns:
        model: 训练好的模型
        train_losses: 训练损失历史
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    
    # 记录训练损失
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
            # 计算损失
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
        
        # 更新学习率
        scheduler.step()
    
    print("训练完成!")
    return model, train_losses


def evaluate_model(model, data_loader):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
    
    Returns:
        predictions: 所有样本的预测结果
        ground_truth: 所有样本的真实标签
        mse: 每个输出维度的均方误差
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_ground_truth = []
    
    with torch.no_grad():
        for features, labels in data_loader:
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
    绘制训练损失曲线
    
    Args:
        train_losses: 训练损失历史
        save_path: 保存图片的路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
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
    output_names = ["x", "y"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for i in range(2):
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
    parser = argparse.ArgumentParser(description='训练球位置预测模型')
    parser.add_argument('--data_dir', type=str, default="processed_data",
                       help='数据目录路径')
    parser.add_argument('--output_dir', type=str, default="checkpoints",
                       help='模型和结果保存目录')
    parser.add_argument('--batch_size', type=int, default=32,
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
    
    # 加载数据集
    dataset = SpikeRefineDataset(args.data_dir, normalize=True)
    
    if len(dataset) == 0:
        print("没有有效数据，退出训练")
        exit()
    
    # 创建数据加载器
    train_loader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 创建模型
    model = SpikeRefineModel()
    
    # 训练模型
    model, train_losses = train_model(
        model, 
        train_loader,
        num_epochs=args.num_epochs, 
        learning_rate=args.learning_rate
    )
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
    }, os.path.join(args.output_dir, "model.pth"))
    
    # 绘制损失曲线
    plot_losses(
        train_losses,
        save_path=os.path.join(args.output_dir, "loss_curve.png")
    )
    
    # 评估模型
    predictions, ground_truth, mse = evaluate_model(model, train_loader)
    print(f"\n均方误差 (MSE):")
    print(f"  x: {mse[0]:.6f}")
    print(f"  y: {mse[1]:.6f}")
    print(f"  平均: {np.mean(mse):.6f}")

    # computer mae
    mae = np.mean(np.abs(predictions - ground_truth), axis=0)
    print(f"\n平均绝对误差 (MAE):")
    print(f"  x: {mae[0]:.6f}")
    print(f"  y: {mae[1]:.6f}")
    print(f"  平均: {np.mean(mae):.6f}")

    # computer rmse
    rmse = np.sqrt(mse)
    print(f"\n均方根误差 (RMSE):")
    print(f"  x: {rmse[0]:.6f}")
    print(f"  y: {rmse[1]:.6f}")
    print(f"  平均: {np.mean(rmse):.6f}")
    
    # 绘制预测值与真实值的对比
    plot_predictions(
        predictions,
        ground_truth,
        save_path=os.path.join(args.output_dir, "predictions_vs_ground_truth.png")
    )
    
    print(f"\n模型和结果已保存到: {args.output_dir}")