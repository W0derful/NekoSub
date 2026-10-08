use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub audio: AudioConfig,
    pub asr: AsrConfig,
    pub vad: VadConfig,
    pub translate: TranslateConfig,
    pub display: DisplayConfig,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    /// 音频后端：pulseaudio / pipewire / alsa，留空自动选择
    pub host: Option<String>,
    /// 输入设备子串，留空自动选择系统音频 monitor 源
    pub device: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AsrConfig {
    /// Whisper GGML 模型路径
    pub model: String,
    /// Silero VAD GGML 模型路径
    pub vad_model: String,
    /// 识别源语言（ja / en / auto ...）
    pub language: String,
    /// Whisper 推理线程数，留空自动
    pub threads: Option<u32>,
    /// GPU 设备号（仅在编译了 vulkan/cuda 后端时生效）
    pub gpu_device: Option<i32>,
    /// 强制 CPU 推理
    pub cpu: bool,
    /// 每积累多少秒音频做一次识别
    pub process_interval_sec: f64,
    /// 单句最长秒数；连续说话超过后强制断句（0 = 不限制）
    pub max_sentence_sec: f64,
    /// 禁用 Whisper 的上文提示（等效 whisper.cpp no_context）：
    /// 每句独立解码，抑制复读自我强化，但术语/人名跨句一致性会下降
    pub no_context: bool,
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            model: "~/.local/share/nekosub/models/ggml-medium.bin".to_string(),
            vad_model: "~/.local/share/nekosub/models/ggml-silero-v5.1.2.bin".to_string(),
            language: "ja".to_string(),
            threads: None,
            gpu_device: None,
            cpu: false,
            process_interval_sec: 1.0,
            max_sentence_sec: 20.0,
            no_context: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(default)]
pub struct VadConfig {
    /// 语音概率阈值
    pub threshold: f32,
    /// 最短语音段（毫秒）
    pub min_speech_ms: i32,
    /// 句尾静音判定时长（毫秒），越低延迟越小
    pub min_silence_ms: i32,
    /// 静音多久后重置识别状态（秒）
    pub silence_reset_sec: f64,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech_ms: 250,
            min_silence_ms: 500,
            // 字幕场景 0.5 秒停顿即视为句尾；过长会让句子连绵不断并诱发复读
            silence_reset_sec: 0.5,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct TranslateConfig {
    /// 翻译后端预设：deepseek / qwen / doubao / openai / ollama / mimo / generic
    pub provider: String,
    /// API 协议：openai / anthropic
    pub protocol: String,
    /// 目标语言（zh / en / ...）
    pub to: String,
    /// API Key；推荐改用环境变量（见 README）
    pub api_key: Option<String>,
    /// 覆盖预设的 OpenAI 兼容 Base URL
    pub base_url: Option<String>,
    /// 翻译模型名；留空用预设默认
    pub model: Option<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    /// 附带的上下文句数（原文/译文对）；0 = 不带上下文。
    /// 断句较碎时需要更多上下文保证翻译连贯
    pub context_sentences: usize,
    /// 自定义 system prompt，支持 {from} / {to} 占位符
    pub prompt: Option<String>,
}

impl Default for TranslateConfig {
    fn default() -> Self {
        Self {
            provider: "deepseek".to_string(),
            protocol: "openai".to_string(),
            to: "zh".to_string(),
            api_key: None,
            base_url: None,
            model: None,
            max_tokens: 512,
            temperature: 0.2,
            context_sentences: 10,
            prompt: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct DisplayConfig {
    pub timestamps: bool,
    pub tentative: bool,
    /// 翻译记录保存目录（按日期存为 txt），留空不保存
    pub save_dir: Option<String>,
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            timestamps: true,
            tentative: true,
            save_dir: None,
        }
    }
}

pub fn default_config_path() -> PathBuf {
    if let Some(dir) = std::env::var_os("XDG_CONFIG_HOME") {
        return Path::new(&dir).join("nekosub").join("config.toml");
    }
    let home = std::env::var_os("HOME").unwrap_or_default();
    Path::new(&home)
        .join(".config")
        .join("nekosub")
        .join("config.toml")
}

impl Config {
    /// 加载配置文件。`explicit` 为 Some 时文件必须存在；否则缺省时返回默认配置。
    pub fn load(explicit: Option<&Path>) -> Result<Config> {
        let path = match explicit {
            Some(p) => p.to_path_buf(),
            None => default_config_path(),
        };
        if !path.exists() {
            if explicit.is_some() {
                bail!("配置文件不存在：{}", path.display());
            }
            return Ok(Config::default());
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("读取配置文件失败：{}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("解析配置文件失败：{}", path.display()))
    }
}

/// 展开路径开头的 `~`
pub fn expand_tilde(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return Path::new(&home).join(rest);
        }
    }
    PathBuf::from(p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_partial_config() {
        let cfg: Config = toml::from_str(
            r#"
            [asr]
            language = "ja"

            [translate]
            provider = "ollama"
            to = "zh"
            max_tokens = 256
            "#,
        )
        .unwrap();
        assert_eq!(cfg.asr.language, "ja");
        assert_eq!(cfg.translate.provider, "ollama");
        assert_eq!(cfg.translate.max_tokens, 256);
        assert_eq!(cfg.vad.min_silence_ms, 500);
        assert!(cfg.display.tentative);
    }
}
