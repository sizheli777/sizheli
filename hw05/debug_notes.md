# 调试记录 (debug_notes.md)

## 1. 环境配置问题

### 问题1：PyTorch版本兼容性

**现象：**
ImportError: cannot import name 'datasets' from 'torchvision'

text

**原因分析：**
torchvision版本过旧（<0.10.0），某些API在不同版本间有变化。

**解决方案：**
```bash
pip install torch>=1.9.0 torchvision>=0.10.0
