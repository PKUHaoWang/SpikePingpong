import torch
import torch.nn as nn

class SpikeRefineModel(nn.Module):
    def __init__(self, input_dim=21, hidden_dim=64, output_dim=2):
        """
        初始化模型
        Args:
            input_dim: 输入维度 (9 + 9 + 3 = 21)
                - 9: 前三帧位置 (3帧 * 3维)
                - 9: 前三帧速度 (3帧 * 3维)
                - 3: end_position
            hidden_dim: 隐藏层维度
            output_dim: 输出维度 (2: x, y坐标)
        """
        super(SpikeRefineModel, self).__init__()
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, features):
        # 检查输入字典是否包含所有必需的键
        required_keys = ['position', 'velocity', 'end_position']
        for key in required_keys:
            if key not in features:
                raise KeyError(f"Missing key in features: {key}")
        
        # 提取并连接特征
        x = torch.cat([
            features['position'],  # [B, 9]
            features['velocity'],  # [B, 9]
            features['end_position']  # [B, 3]
        ], dim=1)  # [B, 21]
        
        # 特征提取
        x = self.feature_extractor(x)

        # Transformer 编码器
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        
        # 输出预测
        predictions = self.output_layer(x)  # [B, 2]
        return predictions

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    model = SpikeRefineModel().to(device)
    print(f"模型结构:\n{model}")
    
    # 模拟输入数据
    batch_size = 2
    features = {
        'position': torch.randn(batch_size, 9).to(device),      # 前三帧位置
        'velocity': torch.randn(batch_size, 9).to(device),      # 前三帧速度
        'end_position': torch.randn(batch_size, 3).to(device)   # 终点位置
    }
    
    # 前向传播
    predictions = model(features)
    print(f"\n输入形状:")
    print(f"- 位置: {features['position'].shape}")
    print(f"- 速度: {features['velocity'].shape}")
    print(f"- 终点位置: {features['end_position'].shape}")
    print(f"\n输出形状: {predictions.shape}")
    print(f"预测值:\n{predictions}")

if __name__ == "__main__":
    main()

    