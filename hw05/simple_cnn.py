"""
极简卷积神经网络 - 参考自公众号文章《计算机视觉》第10篇
结构: Conv1(1,32,3,1) -> ReLU -> MaxPool(2) -> Conv2(32,64,3,1) -> ReLU -> MaxPool(2) -> FC(1600,128) -> ReLU -> FC(128,10)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """极简CNN，用于MNIST手写数字识别"""
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层1: 输入1通道, 输出32通道, 卷积核3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        # 卷积层2: 输入32通道, 输出64通道, 卷积核3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # MNIST 28x28经过两次池化变成7x7
        self.fc2 = nn.Linear(128, 10)
        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Conv1 -> ReLU -> Pool
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv2 -> ReLU -> Pool
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # 展平
        x = x.view(-1, 64 * 7 * 7)
        
        # FC1 -> ReLU -> Dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # FC2 (输出层)
        x = self.fc2(x)
        
        return x

    def get_param_count(self):
        """返回模型参数量"""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    model = SimpleCNN()
    print("=== 极简CNN结构 ===")
    print(model)
    print(f"\n参数量: {model.get_param_count():,}")
    
    # 测试前向传播
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"输出尺寸: {output.shape}")
