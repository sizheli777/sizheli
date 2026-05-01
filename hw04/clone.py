"""
ASR实验代码：基于Sherpa-ONNX的中文语音识别
支持：音频文件识别 / 麦克风实时识别
"""

import sys
import soundfile as sf
import sherpa_onnx


# ==================== 配置区 ====================

def create_recognizer():
    """
    初始化识别器
    使用 Zipformer CTC 中文模型，CPU推理
    """
    # 模型会自动下载到本地缓存
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        model="./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03",  # 中文CTC模型
        tokens="./tokens.txt",  # 与模型配套的tokens文件
        num_threads=4,  # CPU线程数
        decoding_method="greedy_search",
        provider="cpu",  # CPU推理
    )
    return recognizer


# 若模型尚未下载，Sherpa-ONNX会自动从HuggingFace拉取
# 也可手动下载并指定本地路径：
# 模型: https://huggingface.co/csukuangfj/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03
# 或使用内置API直接下载

def create_recognizer_auto():
    """
    自动下载模型的方式（推荐）
    """
    recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
        model="./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03",
        tokens="./tokens.txt",
        num_threads=4,
        decoding_method="greedy_search",
    )
    return recognizer


def transcribe_file(recognizer, audio_path: str):
    """
    识别音频文件
    """
    print(f"[*] 正在识别音频文件: {audio_path}")

    # 读入音频
    audio, sr = sf.read(audio_path, dtype="float32")

    # 重采样到16kHz（ASR模型输入要求）
    if sr != 16000:
        print(f"[!] 采样率 {sr}Hz -> 16000Hz，请先自行转换或在此处理")
        # 这里简化处理，实际可用librosa等重采样

    # 直接识别全段音频
    result = recognizer.decode(audio, sr)

    print(f"[结果] 采样率: {sr}Hz, 音频长度: {len(audio) / sr:.2f}秒")
    print(f"[识别文本]:\n{result}\n")

    return result


def transcribe_microphone(recognizer, duration_seconds: int = 10):
    """
    从麦克风实时录制并识别
    """
    import sounddevice as sd

    print(f"[*] 准备录音，时长 {duration_seconds} 秒...")
    print("[*] 请在提示后开始说话...")

    samplerate = 16000
    input("按 Enter 开始录音...")

    # 录制音频
    print("[录音中...]")
    audio = sd.rec(
        int(duration_seconds * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="float32"
    )
    sd.wait()  # 等待录音完成
    audio = audio.flatten()

    print("[录音结束] 正在识别...")

    # 识别
    result = recognizer.decode(audio, samplerate)

    print(f"[识别文本]:\n{result}\n")

    return result


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("=" * 50)
    print("ASR 实验 - Sherpa-ONNX + Zipformer CTC (中文)")
    print("=" * 50)

    # 初始化识别器
    print("[*] 初始化识别器（首次运行会下载模型，约100MB）...")
    recognizer = create_recognizer_auto()
    print("[✓] 识别器就绪\n")

    # 模式选择
    mode = input("请选择模式 [1] 文件识别  [2] 麦克风录制: ").strip()

    if mode == "1":
        filepath = input("请输入音频文件路径（如 ./task2_voice.wav）: ").strip()
        transcribe_file(recognizer, filepath)

    elif mode == "2":
        sec = input("请输入录音时长（秒，默认10）: ").strip()
        sec = int(sec) if sec else 10
        transcribe_microphone(recognizer, sec)

    else:
        print("无效选择，退出。")