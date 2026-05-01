"""
一键运行两个模型训练并输出对比结果
"""
import torch
import subprocess
import sys

def run_script(script_name):
    """运行Python脚本"""
    print(f"\n{'='*60}")
    print(f"运行 {script_name}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    return result.returncode == 0

def main():
    print("=" * 60)
    print("hw05 - CNN手写数字识别实验")
    print("运行极简CNN和LeNet-5训练")
    print("=" * 60)
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU不可用，使用CPU训练")
    
    # 运行任务一
    success1 = run_script("train_simple_cnn.py")
    if not success1:
        print("极简CNN训练失败!")
    
    # 运行任务二
    success2 = run_script("train_lenet5.py")
    if not success2:
        print("LeNet-5训练失败!")
    
    print("\n" + "=" * 60)
    print("两个模型训练完成!")
    print("详细对比请查看 report.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
