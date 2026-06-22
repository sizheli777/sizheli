# 📊 AI 数据分析助手

> 基于 Streamlit + LLM 的智能数据分析应用 —— 上传数据，自然语言提问，AI 自动分析并可视化。

## 🎯 项目简介

AI 数据分析助手是一个 Web 应用，让非技术用户通过自然语言就能完成数据探索、统计分析和可视化。用户只需上传 CSV 或 Excel 文件，然后用中文描述分析需求，AI 会自动理解问题、编写并执行 Python 分析代码，最后返回结果和图表。

**适用场景：** 快速数据探索、销售分析、学生成绩分析、问卷调查统计等。

## 📁 目录结构

```
期末作业/
├── app.py                    # Streamlit 主应用入口
├── src/
│   ├── __init__.py           # 包初始化
│   ├── data_loader.py        # 数据加载与概要分析
│   ├── llm_service.py        # LLM API 调用服务
│   ├── code_executor.py      # 安全代码执行引擎
│   └── visualization.py      # 图表渲染与导出
├── demo/
│   └── sample_sales.csv      # 演示用销售数据
├── requirements.txt          # Python 依赖列表
├── .env.example              # 环境变量配置模板
├── README.md                 # 本文件
└── report.md                 # 项目报告
```

## 🚀 一键运行

### 1. 环境准备

```bash
# 克隆或进入项目目录
cd 期末作业

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

支持任何 OpenAI 兼容 API（DeepSeek、OpenAI、Ollama 等）。

**方式一：环境变量（推荐）**
```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env 填入你的 API Key
# API_KEY=sk-your-api-key-here
# BASE_URL=https://api.deepseek.com
```

**方式二：在应用侧边栏直接输入**

### 3. 启动应用

```bash
streamlit run app.py
```

浏览器会自动打开 `http://localhost:8501`

### 4. 开始分析

1. 在左侧边栏输入 API Key 并点击「连接测试」
2. 上传 CSV 或 Excel 数据文件
3. 切换到「智能分析」标签页
4. 输入问题，例如：
   - "各地区的销售额总和是多少？"
   - "画出各地区销售额的柱状图"
   - "分析销售额和利润的相关性"
   - "哪个产品的利润最高？"
   - "按月统计销售额趋势"

## 📸 效果展示

### 数据概览
- 自动展示数据行数、列数、缺失值统计
- 列信息与数值统计一览
- 交互式数据表格

### 智能分析
- 💬 聊天式自然语言交互
- 🧠 AI 解释分析思路
- 💻 展示并执行生成的分析代码
- 📊 自动渲染 matplotlib/seaborn 图表
- 📥 支持图表和对话记录下载

## 🔧 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| UI | Streamlit | 快速构建数据应用界面 |
| 数据处理 | pandas, numpy | 数据分析核心库 |
| 可视化 | matplotlib, seaborn | 图表生成 |
| AI 引擎 | LLM (DeepSeek/OpenAI) | 自然语言理解与代码生成 |
| API 调用 | openai SDK | OpenAI 兼容接口 |

## ⚠️ 注意事项

- **API Key 安全：** 不要将 API Key 提交到公开仓库，使用环境变量或 `.env` 文件
- **代码执行安全：** 应用使用了受限的代码执行环境，但仍建议在受信任的环境中运行
- **网络要求：** 需能访问 LLM API 服务（DeepSeek 或 OpenAI）
- **中文支持：** 图表中文显示需要系统安装中文字体（SimHei 或 Microsoft YaHei）

## 📹 演示视频

*（演示视频链接待补充）*

## 📄 许可

本项目为课程期末作业，仅供学习参考。
