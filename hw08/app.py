"""
AI 数据分析助手 — Streamlit 主应用
通过自然语言交互，自动分析 CSV/Excel 数据并生成可视化
"""

import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# 自动加载 .env 文件中的环境变量
def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, val = line.partition("=")
                    key, val = key.strip(), val.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = val
_load_env()

import streamlit as st
import pandas as pd
import time

from data_loader import load_data, get_data_info, get_data_schema_for_prompt
from llm_service import build_system_prompt, create_client, ask_llm
from code_executor import execute_code, format_code_for_display
from visualization import render_figures, create_figure_download_buttons


# ==================== 页面配置 ====================
st.set_page_config(
    page_title="AI 数据分析助手",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== 样式 ====================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .main-header h1 {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stAlert { border-radius: 10px; }
    .code-block {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Cascadia Code', 'Fira Code', monospace;
        font-size: 0.9rem;
        overflow-x: auto;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 会话状态初始化 ====================
DEFAULT_STATE = {
    "df": None,
    "data_info": None,
    "data_schema_text": "",
    "messages": [],          # 对话历史
    "system_prompt": "",
    "api_configured": False,
    "client": None,
}

for key, default in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ==================== 辅助函数 ====================
def reset_chat():
    """重置对话历史"""
    st.session_state.messages = []


def add_message(role: str, content: str):
    """添加消息到对话历史"""
    st.session_state.messages.append({"role": role, "content": content})


# ==================== 侧边栏 ====================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=64)
    st.title("⚙️ 配置面板")

    # --- API 配置 ---
    st.subheader("🔑 API 配置")

    # 从环境变量读取默认值
    env_api_key = os.environ.get("API_KEY", "")
    env_base_url = os.environ.get("BASE_URL", "https://api.deepseek.com")
    env_model = os.environ.get("MODEL", "deepseek-chat")

    # 初始化默认值
    if "api_key" not in st.session_state:
        st.session_state.api_key = env_api_key
    if "base_url" not in st.session_state:
        st.session_state.base_url = env_base_url
    if "model" not in st.session_state:
        st.session_state.model = env_model

    # 首次自动连接
    if env_api_key and not st.session_state.api_configured and "auto_connect_tried" not in st.session_state:
        st.session_state.auto_connect_tried = True
        try:
            client = create_client(env_api_key, env_base_url)
            client.chat.completions.create(
                model=env_model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
            st.session_state.api_configured = True
            st.session_state.client = client
        except Exception:
            pass  # 自动连接失败，用户可手动配置

    with st.expander("展开配置", expanded=not st.session_state.api_configured):
        api_key = st.text_input(
            "API Key",
            type="password",
            value=st.session_state.api_key,
            placeholder="sk-xxxxxxxxxxxxxxxx",
            help="输入你的 API Key（DeepSeek / OpenAI 兼容）"
        )

        base_url = st.text_input(
            "Base URL",
            value=st.session_state.base_url,
            placeholder="https://api.deepseek.com",
            help="API 基础 URL，支持任何 OpenAI 兼容接口"
        )

        model = st.selectbox(
            "模型",
            options=["deepseek-chat", "deepseek-coder", "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
            index=0,
            help="选择要使用的 LLM 模型"
        )

        if st.button("✅ 连接测试", use_container_width=True):
            if not api_key:
                st.error("请输入 API Key")
            else:
                try:
                    client = create_client(api_key, base_url)
                    test_response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": "Hello"}],
                        max_tokens=10,
                    )
                    st.session_state.api_configured = True
                    st.session_state.client = client
                    st.session_state.api_key = api_key
                    st.session_state.base_url = base_url
                    st.session_state.model = model
                    st.success("✅ 连接成功！")
                except Exception as e:
                    st.error(f"连接失败: {str(e)}")

    st.divider()

    # --- 文件上传 ---
    st.subheader("📁 数据上传")
    uploaded_file = st.file_uploader(
        "上传 CSV 或 Excel 文件",
        type=["csv", "xlsx", "xls"],
        help="支持 CSV (UTF-8/GBK) 和 Excel (.xlsx/.xls) 格式",
        on_change=reset_chat,
    )

    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            st.session_state.df = df
            st.session_state.data_info = get_data_info(df)
            st.session_state.data_schema_text = get_data_schema_for_prompt(df)
            st.session_state.system_prompt = build_system_prompt(
                st.session_state.data_schema_text
            )
            st.success(f"✅ 已加载: {df.shape[0]} 行 × {df.shape[1]} 列")
        except Exception as e:
            st.error(f"文件加载失败: {str(e)}")
            st.session_state.df = None

    st.divider()

    # --- 会话管理 ---
    st.subheader("💬 会话管理")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 清空对话", use_container_width=True):
            reset_chat()
            st.rerun()
    with col2:
        if st.button("🗑️ 全部重置", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    if st.session_state.messages:
        st.caption(f"当前对话: {len(st.session_state.messages) // 2} 轮")

    st.divider()

    # --- 关于 ---
    with st.expander("ℹ️ 关于"):
        st.markdown("""
        **AI 数据分析助手 v1.0**

        上传数据文件，用自然语言提问，
        AI 自动分析并生成可视化。

        **支持的分析类型：**
        - 📊 描述性统计
        - 📈 趋势分析
        - 🔍 数据筛选与查询
        - 📉 相关性分析
        - 🎨 可视化图表
        - 🧹 数据清洗建议
        """)


# ==================== 主区域 ====================
st.markdown('<div class="main-header"><h1>📊 AI 数据分析助手</h1></div>', unsafe_allow_html=True)
st.caption("上传数据 → 自然语言提问 → AI 自动分析并可视化")

# 检查数据是否加载
if st.session_state.df is None:
    st.info("👈 请在左侧边栏上传 CSV 或 Excel 数据文件开始分析")

    # 提供样例数据
    st.markdown("### 💡 没有数据？试试这个")
    if st.button("📥 加载演示数据（销售数据）"):
        sample_data = {
            "日期": pd.date_range("2025-01-01", periods=50, freq="W"),
            "地区": ["华东"] * 10 + ["华南"] * 10 + ["华北"] * 10 + ["西南"] * 10 + ["华中"] * 10,
            "产品": ["产品A", "产品B", "产品C", "产品D", "产品E"] * 10,
            "销售额": [
                15200, 23400, 18700, 32100, 28900, 14500, 25600, 19800, 33400, 27600,
                18300, 22100, 16500, 29800, 31200, 16700, 24300, 19200, 28700, 35400,
                12500, 27800, 21300, 33600, 29100, 13400, 26500, 18700, 34500, 29800,
                19800, 31200, 23400, 35600, 27800, 17600, 28900, 22300, 36700, 30100,
                21100, 28700, 24500, 31200, 33400, 18700, 29800, 25600, 34500, 28700,
            ],
            "利润": [
                3200, 5100, 4300, 7800, 6500, 3100, 5800, 4500, 8200, 6100,
                4100, 4900, 3700, 7200, 7400, 3800, 5500, 4400, 6900, 8500,
                2800, 6400, 4900, 8000, 6700, 3000, 6100, 4200, 8300, 7000,
                4500, 7400, 5400, 8600, 6500, 4000, 6800, 5100, 8900, 7200,
                4800, 6700, 5600, 7400, 8000, 4300, 7100, 5900, 8300, 6800,
            ],
            "数量": [120, 185, 150, 260, 230, 115, 205, 160, 270, 220] * 5,
        }
        demo_df = pd.DataFrame(sample_data)
        st.session_state.df = demo_df
        st.session_state.data_info = get_data_info(demo_df)
        st.session_state.data_schema_text = get_data_schema_for_prompt(demo_df)
        st.session_state.system_prompt = build_system_prompt(st.session_state.data_schema_text)
        st.rerun()

    st.stop()


# ==================== Tab 布局 ====================
tab1, tab2, tab3 = st.tabs(["📋 数据概览", "💬 智能分析", "📜 分析历史"])

# ==================== Tab 1: 数据概览 ====================
with tab1:
    st.subheader("数据预览")
    info = st.session_state.data_info

    # 关键指标
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 数据行数", f"{info['rows']:,}")
    with col2:
        st.metric("📋 数据列数", info["columns"])
    with col3:
        total_missing = sum(info["missing"].values())
        st.metric("❓ 缺失值总数", f"{total_missing:,}")
    with col4:
        numeric_count = len(info.get("numeric_columns", []))
        st.metric("🔢 数值列数", numeric_count)

    st.divider()

    # 数据表格
    st.dataframe(
        st.session_state.df.head(100),
        use_container_width=True,
        height=300,
    )

    st.divider()

    # 列信息
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("列信息")
        col_data = []
        for col_name in info["column_names"]:
            col_data.append({
                "列名": col_name,
                "类型": info["dtypes"].get(col_name, ""),
                "缺失数": info["missing"].get(col_name, 0),
                "缺失率": f"{info['missing_percent'].get(col_name, 0):.1f}%",
            })
        st.dataframe(pd.DataFrame(col_data), use_container_width=True)

    with col_right:
        st.subheader("数值列统计")
        if "numeric_stats" in info:
            st.dataframe(
                pd.DataFrame(info["numeric_stats"]).round(2),
                use_container_width=True,
            )
        else:
            st.info("无数值列")

# ==================== Tab 2: 智能分析 ====================
with tab2:
    # 检查 API 是否配置
    if not st.session_state.api_configured:
        st.warning("⚠️ 请先在左侧边栏配置 API Key 并点击「连接测试」")
    else:
        # 显示历史对话
        chat_container = st.container()
        with chat_container:
            for i, msg in enumerate(st.session_state.messages):
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(msg["content"])
                elif msg["role"] == "assistant":
                    with st.chat_message("assistant"):
                        # 展示助手的回复（可能包含多个部分）
                        parts = msg.get("parts", {})
                        if parts.get("explanation"):
                            with st.expander("🧠 分析思路", expanded=False):
                                st.markdown(parts["explanation"])
                        if parts.get("answer"):
                            st.markdown(parts["answer"])
                        if parts.get("figures"):
                            render_figures(parts["figures"])
                            create_figure_download_buttons(parts["figures"])
                        if parts.get("stdout"):
                            with st.expander("📟 输出日志", expanded=False):
                                st.text(parts["stdout"])
                        if parts.get("code"):
                            with st.expander("💻 查看执行代码", expanded=False):
                                st.code(format_code_for_display(parts["code"]), language="python")

        # 输入区域
        st.divider()
        user_question = st.chat_input("💬 输入你的数据分析问题...")

        if user_question:
            # 显示用户消息
            with st.chat_message("user"):
                st.markdown(user_question)
            add_message("user", user_question)

            # AI 处理
            with st.chat_message("assistant"):
                with st.spinner("🤔 AI 正在分析..."):
                    try:
                        # 调用 LLM
                        response = ask_llm(
                            client=st.session_state.client,
                            model=st.session_state.model,
                            system_prompt=st.session_state.system_prompt,
                            user_question=user_question,
                            conversation_history=st.session_state.messages,
                        )

                        explanation = response.get("explanation", "")
                        code = response.get("code", "")
                        answer = response.get("answer", "")

                        # 展示分析思路
                        if explanation:
                            with st.expander("🧠 分析思路", expanded=False):
                                st.markdown(explanation)

                        # 执行代码
                        stdout_text = ""
                        figures = []
                        exec_error = ""

                        if code and code.strip():
                            with st.expander("💻 执行代码", expanded=True):
                                st.code(format_code_for_display(code), language="python")

                            with st.spinner("⚡ 正在执行分析代码..."):
                                time.sleep(0.3)
                                stdout_text, figures, exec_error = execute_code(
                                    code, st.session_state.df
                                )

                            if exec_error:
                                st.error(f"❌ {exec_error}")
                            elif stdout_text:
                                with st.expander("📟 输出日志", expanded=False):
                                    st.text(stdout_text)
                        elif not answer:
                            st.warning("AI 未生成有效响应，请重试")

                        # 展示文字回答
                        if answer:
                            st.markdown(answer)

                        # 展示图表
                        if figures:
                            render_figures(figures)
                            create_figure_download_buttons(figures)

                        # 保存到历史
                        msg_parts = {
                            "explanation": explanation,
                            "code": code,
                            "answer": answer,
                            "stdout": stdout_text if not exec_error else "",
                            "figures": figures,
                        }
                        add_message("assistant", user_question)
                        # 更新最后一条消息的 parts
                        st.session_state.messages[-1]["parts"] = msg_parts

                    except Exception as e:
                        st.error(f"❌ 分析出错: {str(e)}")

# ==================== Tab 3: 分析历史 ====================
with tab3:
    st.subheader("📜 分析历史记录")

    if not st.session_state.messages:
        st.info("暂无分析记录。在「智能分析」标签页中提问后，记录会显示在这里。")
    else:
        # 统计
        user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
        assistant_msgs = [m for m in st.session_state.messages if m["role"] == "assistant"]
        st.caption(f"共 {len(user_msgs)} 次问答")

        for i, msg in enumerate(st.session_state.messages):
            if msg["role"] == "user":
                with st.expander(f"Q{i // 2 + 1}: {msg['content'][:60]}...", expanded=False):
                    st.markdown(f"**问:** {msg['content']}")
                    # 找到对应的回答
                    next_idx = i + 1
                    if next_idx < len(st.session_state.messages):
                        resp = st.session_state.messages[next_idx]
                        parts = resp.get("parts", {})
                        if parts.get("answer"):
                            st.markdown(f"**答:** {parts['answer']}")
                        if parts.get("explanation"):
                            st.caption(f"思路: {parts['explanation'][:200]}")

    # 导出按钮
    if st.session_state.messages:
        st.divider()
        if st.button("📥 导出对话记录 (Markdown)", use_container_width=True):
            md_content = ["# AI 数据分析助手 — 对话记录\n"]
            for i, msg in enumerate(st.session_state.messages):
                if msg["role"] == "user":
                    md_content.append(f"## Q{i // 2 + 1}: {msg['content']}\n")
                else:
                    parts = msg.get("parts", {})
                    if parts.get("answer"):
                        md_content.append(f"{parts['answer']}\n")
                    if parts.get("explanation"):
                        md_content.append(f"*分析思路: {parts['explanation']}*\n")
                    if parts.get("code"):
                        md_content.append(f"\n```python\n{parts['code']}\n```\n")
                    md_content.append("\n---\n")

            st.download_button(
                label="⬇️ 下载对话记录",
                data="\n".join(md_content),
                file_name="analysis_history.md",
                mime="text/markdown",
            )


# ==================== 页脚 ====================
st.divider()
st.caption("📊 AI 数据分析助手 | 基于 Streamlit + LLM | 期末作业 HW08")
