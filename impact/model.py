import torch
import torch.nn as nn

class RobotActionModel(nn.Module):
    def __init__(self, input_dim_speed=18, hidden_dim=64, input_dim_robot=9, output_dim=3):
        super(RobotActionModel, self).__init__()
        
        # MLP 层处理乒乓球数据
        self.pingpang_mlp = nn.Sequential(
            nn.Linear(input_dim_speed, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # MLP 层处理机械臂数据
        self.robot_mlp = nn.Sequential(
            nn.Linear(input_dim_robot, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, features):
        # 检查输入字典是否包含所有必需的键
        required_keys = ['position', 'velocity', 'end_position', 'target_joint']
        for key in required_keys:
            if key not in features:
                raise KeyError(f"Missing key in features: {key}")
        
        # 提取特征
        pingpang = torch.cat([features['position'], features['velocity']], dim=-1)
        robot = torch.cat([features['end_position'], features['target_joint']], dim=-1)

        # 特征处理
        pingpang_features = self.pingpang_mlp(pingpang)
        robot_features = self.robot_mlp(robot)
        
        # 特征融合
        combined_features = torch.cat([pingpang_features, robot_features], dim=1).unsqueeze(1)
        
        # Transformer 处理
        fused_features = self.transformer_encoder(combined_features)
        
        # 最终预测
        predictions = self.output_layer(fused_features[:, -1, :])
        return predictions

# 主函数
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RobotActionModel().to(device)

    # 模拟输入数据
    batch_size = 8
    features = {
        'position': torch.randn(batch_size, 9).to(device),          # 乒乓球位置
        'velocity': torch.randn(batch_size, 9).to(device),          # 乒乓球速度
        'end_position': torch.randn(batch_size, 3).to(device),      # 机械臂终点位置
        'target_joint': torch.randn(batch_size, 6).to(device)       # 目标关节角度
    }

    # 前向传播
    predictions = model(features)
    print("Predictions shape:", predictions.shape)

if __name__ == "__main__":
    main()

    