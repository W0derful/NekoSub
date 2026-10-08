mod asr;
mod audio;
mod config;
mod display;
mod pipeline;
mod translate;

use anyhow::Result;
use clap::Parser;
use config::Config;
use std::path::PathBuf;

/// NekoSub（猫译幕）— 在终端实时显示「原文 + 译文」的语音翻译字幕
#[derive(Parser)]
#[command(name = "nsub", version, about)]
struct Cli {
    /// 配置文件路径（默认 ~/.config/nekosub/config.toml）
    #[arg(long)]
    config: Option<PathBuf>,

    /// 音频输入设备（id 或名称子串，默认自动选择系统音频 monitor 源）
    #[arg(long)]
    device: Option<String>,

    /// 音频后端：pulseaudio / pipewire / alsa（默认自动）
    #[arg(long)]
    host: Option<String>,

    /// 识别源语言，如 ja / en / auto
    #[arg(long = "from")]
    from_lang: Option<String>,

    /// 翻译目标语言，如 zh / en
    #[arg(long = "to")]
    to_lang: Option<String>,

    /// 翻译后端预设：deepseek / qwen / doubao / openai / ollama / mimo / generic
    #[arg(long)]
    translator: Option<String>,

    /// API 协议：openai / anthropic（默认 openai）
    #[arg(long)]
    protocol: Option<String>,

    /// OpenAI 兼容 API 的 Base URL（覆盖预设，任意兼容端点均可）
    #[arg(long)]
    api_base: Option<String>,

    /// API Key（优先级高于环境变量与配置文件）
    #[arg(long)]
    api_key: Option<String>,

    /// 翻译模型名（覆盖预设）
    #[arg(long)]
    model: Option<String>,

    /// Whisper GGML 模型路径
    #[arg(long)]
    asr_model: Option<PathBuf>,

    /// Silero VAD GGML 模型路径
    #[arg(long)]
    vad_model: Option<PathBuf>,

    /// 强制 CPU 推理（不使用 GPU 后端）
    #[arg(long)]
    cpu: bool,

    /// GPU 设备号（编译了 vulkan / cuda 后端时生效）
    #[arg(long)]
    gpu_device: Option<i32>,

    /// 隐藏识别中的临时行
    #[arg(long)]
    no_tentative: bool,

    /// 不显示时间戳
    #[arg(long)]
    no_timestamps: bool,

    /// 列出可用输入设备后退出
    #[arg(long)]
    list_devices: bool,

    /// 输出 Whisper 详细日志（默认静默）
    #[arg(short, long)]
    verbose: bool,
}

fn apply_overrides(cfg: &mut Config, cli: &Cli) -> Result<()> {
    if let Some(v) = &cli.host {
        cfg.audio.host = Some(v.clone());
    }
    if let Some(v) = &cli.device {
        cfg.audio.device = Some(v.clone());
    }
    if let Some(v) = &cli.from_lang {
        cfg.asr.language = v.clone();
    }
    if let Some(v) = &cli.to_lang {
        cfg.translate.to = v.clone();
    }
    if let Some(v) = &cli.translator {
        cfg.translate.provider = v.clone();
    }
    if let Some(v) = &cli.protocol {
        cfg.translate.protocol = v.clone();
    }
    if let Some(v) = &cli.api_base {
        cfg.translate.base_url = Some(v.clone());
    }
    if let Some(v) = &cli.api_key {
        cfg.translate.api_key = Some(v.clone());
    }
    if let Some(v) = &cli.model {
        cfg.translate.model = Some(v.clone());
    }
    if let Some(v) = &cli.asr_model {
        cfg.asr.model = v.display().to_string();
    }
    if let Some(v) = &cli.vad_model {
        cfg.asr.vad_model = v.display().to_string();
    }
    if cli.cpu {
        cfg.asr.cpu = true;
    }
    if let Some(n) = cli.gpu_device {
        cfg.asr.gpu_device = Some(n);
    }
    if cli.no_tentative {
        cfg.display.tentative = false;
    }
    if cli.no_timestamps {
        cfg.display.timestamps = false;
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if cli.list_devices {
        return audio::print_devices(cli.host.as_deref());
    }

    if !cli.verbose {
        // 静默 whisper/ggml 日志；--verbose 时保留其默认 stderr 输出
        yamabiko_whisper::install_log_hooks();
    }

    let mut cfg = Config::load(cli.config.as_deref())?;
    apply_overrides(&mut cfg, &cli)?;

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(pipeline::run(&cfg, cli.api_key.as_deref()))
}
