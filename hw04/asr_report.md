一、三种开源 ASR 方案对比分析
1.1 方案概览
我选取了三个具有代表性且社区活跃的开源 ASR 项目进行比较：OpenAI Whisper、Sherpa-ONNX、FunASR。它们分别代表了通用大规模预训练模型、高性能推理引擎、工业级端到端方案三条不同技术路线。

1.2 详细对比
维度	OpenAI Whisper	Sherpa-ONNX	FunASR
版本/仓库	openai/whisper v20231117 (large-v3) 	k2-fsa/sherpa-onnx v1.12.40 	FunAudioLLM/Fun-ASR (持续更新) 
开源协议	MIT License 	Apache 2.0	CC BY 4.0 (模型) / 代码开源
语言支持	99种语言，包括中英日韩法德等，中文识别优秀	通过集成多种模型支持：Zipformer CTC (中文)、SenseVoice (中粤英日韩)、Paraformer (中英+方言) 等 	中文(含粤语/吴语/闽南语/客家话等方言+各地官话)、英、日、韩、越南语、印尼语、泰语等 
模型体量	5档：tiny(39M)→large-v3(1.55B)。推荐 base(74M) 或 small(244M) 平衡速度与精度 	模型多样：SenseVoice约200MB、Zipformer CTC中文约100MB、Paraformer约300MB 	Fun-ASR-nano（编码器0.2B+解码器0.6B）、Fun-ASR（编码器0.7B+解码器7B）
推理速度	较大模型在CPU上较慢，large模型需GPU才能实时。实测base模型CPU约50ms/秒音频	专为CPU优化，支持INT8量化。骁龙865平台延迟<100ms，CPU占用率较同类低37% 	nano版本针对低资源场景优化；可通过阿里云API调用，本地部署需较高算力 
流式/实时	不原生支持。需完整音频输入后一次性转录	原生支持流式，Chunk-based增量解码、可实时显示中间结果 	支持流式，并专门针对噪声鲁棒性、中英混杂做了优化 
部署难度	⭐⭐简单（pip install openai-whisper + ffmpeg）	⭐⭐中等（pip包完善，模型按需下载）	⭐⭐⭐ 较高（大模型版本需GPU，nano版对硬件要求适中）
依赖与环境	Python 3.8+, PyTorch/torch, ffmpeg	Python 3.6+, numpy, sherpa-onnx（无需PyTorch依赖）	Python 3.8+, PyTorch, modelscope或阿里云SDK
特色功能	支持翻译（语音→英语文本）、语种检测、时间戳	同时集成TTS、VAD、说话人识别/日志、关键词检出、语音增强等8大模块 	热词定制化、远场VAD优化、LLM集成提升语义理解 
平台支持	Linux/macOS/Windows，可Docker部署	覆盖最广：x86/ARM/RISC-V/HarmonyOS，Android/iOS/WebAssembly 	云端API跨平台；本地部署支持Linux/macOS
1.3 Vosk 与 Coqui STT 补充说明
Vosk（v0.3.43，Apache 2.0）：轻量级离线方案，模型约50MB，支持20+语言。优势在于极低资源消耗，适合树莓派等嵌入式设备。但其采用的是较老的Kaldi架构，中文识别准确率低于Whisper和FunASR。

Coqui STT（v1.4.0，MPL-2.0）：前身为Mozilla DeepSpeech，主要面向英文。虽然支持自定义训练，但官方中文模型缺失，社区活跃度下降明显，最后一个正式版本为v1.4.0。

二、选型理由
最终选择：Sherpa-ONNX

理由如下：

任务匹配度高：课程任务要求使用“任务二导出的配音音频”做识别。任务二生成的是中文AI配音，Sherpa-ONNX提供的Zipformer CTC中文模型和SenseVoice多语种模型在中文识别上表现出色，且特别支持吴语、闽南语等多种方言。

流式支持：任务鼓励尝试“麦克风实时输入”，Sherpa-ONNX原生支持流式识别，代码示例清晰，可同时满足音频文件和麦克风两种输入方式。

CPU友好：无需GPU即可流畅运行，笔记本实测推理速度快、内存占用低。INT8量化模型进一步降低了CPU压力。

无PyTorch依赖：纯ONNX Runtime推理，避免了PyTorch庞大的安装体积（~2GB），环境配置更轻量。

未来扩展：Sherpa-ONNX还集成了VAD（语音活动检测）、说话人日志等模块，后续实验可以玩更多花样。

备选方案说明：若不选择Sherpa-ONNX，Whisper的small模型也是成熟的中文识别方案，适合对翻译或时间戳有需求的场景；FunASR-nano则适合追求极致识别精度且有GPU的场景。
