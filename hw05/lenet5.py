"""
LeNet-5 实现 - 用于MNIST手写数字识别
原始LeNet-5结构适配MNIST (32x32输入 -> 28x28输入调整)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    """
    LeNet-5 网络结构
    
    输入: 1x28x28 (MNIST灰度图)
    
    各层参数:
    ┌──────────────┬────────────┬────────────┬─────────────────┐
    │     层名     │ 输入尺寸   │ 输出尺寸   │     参数量       │
    ├──────────────┼────────────┼────────────┼─────────────────┤
    │  Conv1       │ 1x28x28    │ 6x28x28    │ (1*6*5*5)+6=156  │
    │  AvgPool1    │ 6x28x28    │ 6x14x14    │ 0               │
    │  Conv2       │ 6x14x14    │ 16x10x10   │ (6*16*5*5)+16=2416│
    │  AvgPool2    │ 16x10x10   │ 16x5x5     │ 0               │
    │  Conv3 (可选)│ 16x5x5     │ 120x1x1    │ (16*120*5*5)+120=48120│
    │  FC1         │ 120        │ 84         │ 120*84+84=10164  │
    │  FC2         │ 84         │ 10         │ 84*10+10=850     │
    └──────────────┴────────────┴────────────┴─────────────────┘
    总参数量: 约 61,706
    """
    
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        # 卷积层1: 输入1通道(灰度), 输出6通道, 卷积核5x5
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)  # 28-5+1=24? 等等需要调整
        # 实际上原始LeNet-5输入32x32，输出28x28，对于28x28输入需要padding=2保持输出28x28
        
        # 调整为适配28x28输入: padding=2 使输出28x28
        self.conv1_adj = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)  # 28->28
        
        # 卷积层2: 输入6通道, 输出16通道, 卷积核5x5
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        
        # 卷积层3: 输入16通道, 输出120通道, 卷积核5x5 (原始LeNet-5中的C3-F6连接层)
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        
        # 全连接层1: 120 -> 84
        self.fc1 = nn.Linear(120, 84)
        
        # 全连接层2: 84 -> 10
        self.fc2 = nn.Linear(84, num_classes)
        
        # 池化层 (使用平均池化，符合原始LeNet-5)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # C1: 卷积层 -> 激活 -> 池化
        x = self.conv1_adj(x)      # 1x28x28 -> 6x28x28
        x = F.tanh(x)               # 原始LeNet使用tanh
        x = self.pool(x)            # 6x28x28 -> 6x14x14
        
        # C3: 卷积层 -> 激活 -> 池化
        x = self.conv2(x)           # 6x14x14 -> 16x10x10 (14-5+1=10)
        x = F.tanh(x)
        x = self.pool(x)            # 16x10x10 -> 16x5x5
        
        # C5: 卷积层 (相当于全连接)
        x = self.conv3(x)           # 16x5x5 -> 120x1x1 (5-5+1=1)
        x = F.tanh(x)
        
        # 展平
        x = x.view(-1, 120)
        
        # F6: 全连接层
        x = self.fc1(x)             # 120 -> 84
        x = F.tanh(x)
        
        # 输出层
        x = self.fc2(x)             # 84 -> 10
        
        return x
    
    def forward_with_logits(self, x):
        """返回logits和中间层输出，便于分析"""
        features = []
        
        x = self.conv1_adj(x)
        x = F.tanh(x)
        features.append(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = F.tanh(x)
        features.append(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = F.tanh(x)
        
        x = x.view(-1, 120)
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        
        return x, features
    
    def get_param_count(self):
        """返回模型参数量"""
        return sum(p.numel() for p in self.parameters())


# 另一种更接近原始LeNet-5(32x32输入)的版本
class LeNet5Original(nn.Module):
    """原始LeNet-5 (输入32x32，需对MNIST进行resize)"""
    
    def __init__(self, num_classes=10):
        super(LeNet5Original, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)   # 32->28
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)                  # 28->14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)  # 14->10
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)                  # 10->5
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0) # 5->1
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = F.tanh(self.conv1(x))
        x = self.pool1(x)
        x = F.tanh(self.conv2(x))
        x = self.pool2(x)
        x = F.tanh(self.conv3(x))
        x = x.view(-1, 120)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = LeNet5()
    print("=== LeNet-5 结构 ===")
    print(model)
    print(f"\n参数量: {model.get_param_count():,}")
    
    # 测试前向传播
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"输出尺寸: {output.shape}")
    
    # 打印详细参数表
    print("\n=== 详细参数表 ===")
    print("-" * 50)
    print(f"{'层名':<15} {'输入尺寸':<15} {'输出尺寸':<15} {'参数量':<12}")
    print("-" * 50)
    
    # 模拟前向传播计算各层尺寸
    x = torch.randn(1, 1, 28, 28)
    print(f"{'Input':<15} {str(tuple(x.shape)):<15} - {'0':<12}")
    
    x = model.conv1_adj(x)
    print(f"{'Conv1':<15} {str(tuple(x.shape)):<15} {str(tuple(x.shape)):<15} {model.conv1_adj.weight.numel() + model.conv1_adj.bias.numel():<12}")
    
    x = model.pool(x)
    print(f"{'Pool1':<15} {str(tuple(x.shape)):<15} {str(tuple(x.shape)):<15} {'0':<12}")
    
    x = model.conv2(x)
    print(f"{'Conv2':<15} {str(tuple(x.shape)):<15} {str(tuple(x.shape)):<15} {model.conv2.weight.numel() + model.conv2.bias.numel():<12}")
    
    x = model.pool(x)
    print(f"{'Pool2':<15} {str(tuple(x.shape)):<15} {str(tuple(x.shape)):<15} {'0':<12}")
    
    x = model.conv3(x)
    print(f"{'Conv3':<15} {str(tuple(x.shape)):<15} {str(tuple(x.shape)):<15} {model.conv3.weight.numel() + model.conv3.bias.numel():<12}")
    
    x = x.view(-1, 120)
    x = model.fc1(x)
    print(f"{'FC1':<15} {f'({x.shape[0]}, 120)':<15} {f'({x.shape[0]}, 84)':<15} {model.fc1.weight.numel() + model.fc1.bias.numel():<12}")
    
    x = model.fc2(x)
    print(f"{'FC2':<15} {f'({x.shape[0]}, 84)':<15} {f'({x.shape[0]}, 10)':<15} {model.fc2.weight.numel() + model.fc2.bias.numel():<12}")
