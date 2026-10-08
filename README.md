# NekoSub（猫译幕）

在终端实时显示「原文 + 译文」对照字幕的命令行工具。看日语视频/直播时，
捕获系统音频 → 本地 Whisper 流式识别（内置 Silero VAD 切句）→ 调用
OpenAI 兼容 API 翻译 → 终端滚动显示。

```text
[00:12:03] 原文：今日はいい天気ですね
[00:12:03] 译文：今天天气真好啊
```

- 音频捕获与语音识别全部在本地完成，只把识别出的文本发给翻译 API
- 翻译后端可自由选择：任意 OpenAI 兼容端点（`/chat/completions`）都能接入
- 识别中的句子以灰色临时行实时刷新，确认后打印最终的原文/译文

## 构建依赖

| 发行版 | 命令 |
|---|---|
| Arch | `sudo pacman -S clang cmake pkgconf pipewire` |
| Debian/Ubuntu | `sudo apt install clang cmake pkg-config libasound2-dev libpulse-dev` |
| Fedora | `sudo dnf install clang cmake pkgconf alsa-lib-devel pulseaudio-libs-devel` |

> 说明：`clang` 提供 bindgen 需要的 libclang 与内置头文件；`cmake` 用于编译
> whisper.cpp。PulseAudio 开发库用于 cpal 的 `pulseaudio` 后端——PipeWire
> 系统经 pipewire-pulse 兼容层同样可用，无需额外后端。

```bash
cargo build --release                 # CPU 推理
cargo build --release --features vulkan   # GPU 加速（NVIDIA/AMD，需 Vulkan SDK）
cargo build --release --features cuda     # GPU 加速（NVIDIA，需 CUDA Toolkit）
```

## 模型文件

识别使用 whisper.cpp 的 GGML 模型，VAD 使用 GGML 格式的 Silero 模型，均不随程序分发：

```bash
mkdir -p ~/.local/share/nekosub/models
cd ~/.local/share/nekosub/models

# 日语推荐：kotoba-whisper-v2.2（日语微调，蒸馏架构解码快，1.5GB）
curl -LO https://hf-mirror.com/Pomni/kotoba-whisper-v2.2-ggml-allquants/resolve/main/ggml-kotoba-v2.2-f16.bin
# 多语言场景：large-v3-turbo（通用，1.6GB）
curl -LO https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin

# Silero VAD 模型
curl -LO https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin
```

（Hugging Face 直连不通时把域名换成 `hf-mirror.com`。）

### 模型怎么选

| 模型 | 大小 | 显存 | 特点 |
|---|---|---|---|
| `ggml-kotoba-v2.2-f16.bin` | 1.5GB | ~2GB | **日语首选**：日语微调（ReazonSpeech），蒸馏架构解码最快；仅日语 |
| `ggml-large-v3-turbo.bin` | 1.6GB | ~2.2GB | **多语言首选**：通用精度好，解码快（4 层解码器） |
| `ggml-large-v3-q5_0.bin` | 1.1GB | ~1.5GB | turbo 的更小量化档 |
| `ggml-large-v3.bin` | 3.1GB | ~3.5GB | 通用最高精度，但解码器 32 层，**流式识别在中端卡上可能跟不上实时** |
| `ggml-medium.bin` | 1.5GB | ~2GB | 起步选择，日语专有名词较弱 |

要点：

- **流式识别每秒都要重新解码滚动缓冲，解码速度比离线精度数字更重要**——
  优先选蒸馏/turbo 架构（解码器 2～4 层），large-v3 全量（32 层）离线转写强但流式吃力。
- **量化命名**：whisper.cpp 官方量化是 `q5_0` / `q8_0` 等；`q5_k_m` 这类 llama.cpp
  命名官方仓库不存在（个别社区转换仓才有 k 系量化），照抄文件名前先查仓库文件列表。
- **量化与显存**：RTX 2060 6GB 这类显卡跑全量 turbo/kotoba 绰绰有余（约 2GB），
  不必为省显存牺牲精度；q8_0 是精度几乎无损的轻量档。
- **社区转换来源**：kotoba-whisper 的 GGML 由社区转换（官方只发了 v1.0/v2.0 的
  GGML），转换流程标准、风险低，在意的话可换官方 `kotoba-tech/kotoba-whisper-v2.2`
  自行转换或用官方 `kotoba-whisper-v2.0-ggml`。
- **日语专用模型只适合日语**：kotoba / whisper-ja-anime 等微调模型识别其他语言会
  明显变差，多语言内容请换回 large-v3-turbo。
- 动漫域还有 `whisper-ja-anime-v0.3`（turbo 架构微调），但没有现成 GGML 需自行
  转换，且泛化性不如 kotoba。

## 翻译 API

任何 OpenAI 兼容端点都可以用，内置预设：

| `--translator` | 端点 | 默认模型 | Key 环境变量 |
|---|---|---|---|
| `deepseek` | `https://api.deepseek.com/v1` | `deepseek-chat` | `DEEPSEEK_API_KEY` |
| `qwen` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` | `DASHSCOPE_API_KEY` |
| `doubao` | `https://ark.cn-beijing.volces.com/api/v3` | 需 `--model` 指定接入点 ID | `ARK_API_KEY` |
| `openai` | `https://api.openai.com/v1` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| `ollama` | `http://localhost:11434/v1` | `qwen2.5:7b` | 不需要 |
| `mimo` | `https://token-plan-cn.xiaomimimo.com/v1` | 需 `--model` 指定 | `MIMO_API_KEY` |
| `generic` | 需 `--api-base` 指定 | 需 `--model` | `NEKOSUB_API_KEY` |

接入其他服务（Kimi、智谱、硅基流动、LM Studio 等）直接用 `generic`：

```bash
export NEKOSUB_API_KEY=sk-xxx
nsub --translator generic \
     --api-base https://api.moonshot.cn/v1 \
     --model moonshot-v1-8k
```

也可以完全绕过预设，用 `--api-base` 覆盖任意预设的端点。

### Anthropic 协议端点

只有 Claude 协议（`/v1/messages`）的服务用 `--protocol anthropic`，
鉴权自动走 `x-api-key` 头：

```bash
export MIMO_API_KEY=你的key
nsub --translator mimo --protocol anthropic \
     --api-base https://token-plan-cn.xiaomimimo.com/anthropic \
     --model 服务商提供的模型名
```

**API Key 优先级**：`--api-key` 参数 > 环境变量 > 配置文件。
推荐使用环境变量，避免明文落盘。

## 配置文件

默认读取 `~/.config/nekosub/config.toml`（全部字段可选）：

```toml
[audio]
# host = "pulseaudio"          # 音频后端：pulseaudio / alsa，默认自动
# device = ""                  # 输入设备子串，默认自动选系统音频 monitor 源

[asr]
model = "~/.local/share/nekosub/models/ggml-kotoba-v2.2-f16.bin"  # 日语专用
# model = "~/.local/share/nekosub/models/ggml-large-v3-turbo.bin"  # 多语言换这个
vad_model = "~/.local/share/nekosub/models/ggml-silero-v5.1.2.bin"
language = "ja"                # 识别源语言（ja / en / auto）
# threads = 8                  # Whisper 推理线程数
# gpu_device = 0               # GPU 设备号（vulkan/cuda 后端）
# cpu = false                  # true 则强制 CPU
process_interval_sec = 1.0     # 每积累多少秒音频识别一次
max_sentence_sec = 20          # 单句最长秒数，连续说话超时强制断句（0 不限制）
# no_context = true            # 每句独立解码，抑制复读（术语一致性略降）

[vad]
threshold = 0.5                # 语音概率阈值，嘈杂环境可降至 0.4
min_speech_ms = 250            # 最短语音段
min_silence_ms = 500           # 句内停顿容忍
silence_reset_sec = 0.5        # 停顿多久判定句尾（越小断句越快）

[translate]
provider = "deepseek"          # deepseek / qwen / doubao / openai / ollama / mimo / generic
protocol = "openai"            # api 协议：openai / anthropic
to = "zh"                      # 翻译目标语言
# base_url = ""                # 覆盖预设端点
# model = "deepseek-chat"      # 覆盖预设模型
# api_key = ""                 # 不推荐，优先用环境变量
max_tokens = 512
temperature = 0.2
context_sentences = 10         # 附带的上下文句数（0 = 不带上下文；断句碎时保证翻译连贯）

[display]
timestamps = true
tentative = true               # 识别中的灰色临时行
save_dir = "~/Code/NekoSub/NekoSub/text"  # 翻译记录按日期存为 txt，留空不保存
```

## 用法

```bash
nsub                                # 默认：系统音频，日语→中文，deepseek
nsub --list-devices                 # 列出可用输入设备
nsub --from ja --to zh --translator qwen
nsub --translator mimo --model MiMo-xxx   # 小米 MiMo（OpenAI 协议端点）
nsub --device hdmi --gpu-device 1   # 指定采集设备 / GPU
nsub --no-tentative --no-timestamps
nsub --config ./my.toml
```

运行后播放视频/直播即可，`Ctrl-C` 停止。识别中的句子以灰色临时行刷新，
句子确认后按语音顺序打印原文与译文（翻译较慢时自动排队，保证字幕不乱序）。

## 延迟

端到端约 1.5～2.5 秒：句尾静音判定（约 0.5s）+ ASR 推理（约 0.2～0.5s）
+ 翻译 API（约 0.3～0.8s）。可通过调低 `min_silence_ms`、减小
`process_interval_sec` 以及使用 GPU 后端进一步压缩。

## 注意事项

**配置相关**

- 配置只写在 `~/.config/nekosub/config.toml`（或 `--config` 指定的文件）。
  **不要写进 `Cargo.toml`**——那是 Rust 构建清单，程序不读它，写了只会出警告。
- **改配置不需要编译**，重启 `nsub` 即生效；只有改了 Rust 源码才需要
  `cargo build --release --features vulkan`。
- API Key 用环境变量（`DEEPSEEK_API_KEY` 等），不要写进配置文件，避免误提交进 git。

**识别与字幕质量**

- **复读（一句话反复出现）**：Whisper 流式解码在长缓冲下的固有现象。三板斧——
  `vad.silence_reset_sec` 调到 0.8 左右（日语 0.5～1.0，**不要低于 0.5**，太小会把
  句子切得七零八落反而加剧复读）；`asr.no_context = true`（每句独立解码，抑制
  复读的自我强化）；`asr.max_sentence_sec = 12`（超时强制断句兜底）。
- **`no_context` 是双刃剑**：抑制复读的同时，术语/人名的译法可能前后不一致
  （翻译侧的上下文能弥补一部分）。觉得术语乱跳就把 `no_context` 改回 `false`。
- **`temperature` / `temperature_inc` 无法配置**：底层 yamabiko-whisper 未开放这两个
  参数（whisper 的温度回退针对的是"解码失败"，也不是"检测重复"），配置里写了不生效。
- **断句松紧权衡**：`silence_reset_sec` 越小断句越勤、字幕越碎；越大越连贯、
  越容易复读。日语建议 0.8 起步再微调。

**翻译与上下文**

- 翻译请求会附带最近 `context_sentences` 句「原文/译文」对作为上文（建议 10 左右；
  断句碎时上下文能防止代词、话题误判）。**只上传识别出的文字，不上传音频**。
- 上下文越多越准，但 token 消耗线性增长；个人使用 10～20 句都没问题。

**记录落盘**

- `display.save_dir` 按日期生成 txt（如 `text/2026-10-08.txt`），同日多次运行追加写入
  并插入「会话开始」分隔线；翻译失败的句子也带标记保存；留空则不保存。

## 故障排查

- **找不到 monitor 源**：`nsub --list-devices` 查看；用 `--device <子串>` 指定。
- **`401 Unauthorized`**：API Key 未设置或错误，检查对应环境变量。
- **`404 model not found`**：`--model` 名称不对（豆包需要填推理接入点 ID）。
- **构建报 libclang / cmake 错误**：安装 `clang` 与 `cmake`（见上文构建依赖）。
- **识别慢**：用 `--features vulkan/cuda` 重新编译，或换更小的模型（`ggml-small.bin`）。
- **字幕出现复读/超长句**：见上方「注意事项」的复读三板斧；
  仍异常时把 `vad`/`asr` 配置和运行日志发出来看。
