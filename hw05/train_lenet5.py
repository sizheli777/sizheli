"""
训练LeNet-5模型
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import time
import os

from lenet5 import LeNet5


def load_mnist(batch_size=64, data_dir='./data'):
    """加载MNIST数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in tqdm(train_loader, desc='Training'):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    return total_loss / len(train_loader), 100. * correct / total


def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return test_loss / len(test_loader), 100. * correct / total


def main():
    # 超参数配置
    BATCH_SIZE = 64
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 50)
    print("任务二：LeNet-5训练")
    print("=" * 50)
    print(f"设备: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print("-" * 50)
    
    # 加载数据
    print("加载MNIST数据集...")
    train_loader, test_loader = load_mnist(batch_size=BATCH_SIZE)
    
    # 创建模型
    model = LeNet5().to(DEVICE)
    print(f"\n模型参数量: {model.get_param_count():,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练记录
    best_acc = 0
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_lenet5.pth')
            print(f"-> 保存最佳模型 (Acc: {best_acc:.2f}%)")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print("训练完成!")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"最佳测试准确率: {best_acc:.2f}%")
    print("=" * 50)
    
    return {
        'model': model,
        'best_acc': best_acc,
        'total_time': total_time,
        'param_count': model.get_param_count(),
        'train_losses': train_losses,
        'test_accs': test_accs
    }


if __name__ == "__main__":
    results = main()
