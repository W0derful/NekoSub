use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};

/// 最近的翻译记录，作为上下文附给 API。
#[derive(Debug, Clone)]
pub struct Turn {
    pub source: String,
    pub target: String,
}

/// OpenAI 兼容后端预设。
pub struct Preset {
    pub base_url: &'static str,
    pub model: &'static str,
    /// 依次尝试的 API Key 环境变量名
    pub key_env: &'static [&'static str],
    /// 无 key 也能跑（如本地 Ollama）
    pub key_optional: bool,
}

/// API 协议方言。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    /// OpenAI：POST {base}/chat/completions，Bearer 鉴权
    OpenAi,
    /// Anthropic：POST {base}/v1/messages，x-api-key 鉴权
    Anthropic,
}

impl Protocol {
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "openai" => Ok(Self::OpenAi),
            "anthropic" | "claude" => Ok(Self::Anthropic),
            other => bail!("未知 API 协议 \"{other}\"，可选：openai / anthropic"),
        }
    }
}

pub fn preset(name: &str) -> Option<Preset> {
    match name.to_ascii_lowercase().as_str() {
        "deepseek" => Some(Preset {
            base_url: "https://api.deepseek.com/v1",
            model: "deepseek-chat",
            key_env: &["DEEPSEEK_API_KEY"],
            key_optional: false,
        }),
        "qwen" | "dashscope" => Some(Preset {
            base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
            model: "qwen-plus",
            key_env: &["DASHSCOPE_API_KEY"],
            key_optional: false,
        }),
        // 豆包（火山方舟）：model 需填你创建的推理接入点 ID
        "doubao" | "ark" => Some(Preset {
            base_url: "https://ark.cn-beijing.volces.com/api/v3",
            model: "",
            key_env: &["ARK_API_KEY"],
            key_optional: false,
        }),
        "openai" => Some(Preset {
            base_url: "https://api.openai.com/v1",
            model: "gpt-4o-mini",
            key_env: &["OPENAI_API_KEY"],
            key_optional: false,
        }),
        "ollama" => Some(Preset {
            base_url: "http://localhost:11434/v1",
            model: "qwen2.5:7b",
            key_env: &[],
            key_optional: true,
        }),
        // 小米 MiMo（token-plan）：OpenAI 协议端点，模型名以服务商提供为准
        "mimo" => Some(Preset {
            base_url: "https://token-plan-cn.xiaomimimo.com/v1",
            model: "",
            key_env: &["MIMO_API_KEY"],
            key_optional: false,
        }),
        "generic" => Some(Preset {
            base_url: "",
            model: "",
            key_env: &["NEKOSUB_API_KEY"],
            key_optional: true,
        }),
        _ => None,
    }
}

pub fn providers() -> &'static [&'static str] {
    &[
        "deepseek",
        "qwen",
        "doubao",
        "openai",
        "ollama",
        "mimo",
        "generic",
    ]
}

/// 按优先级解析 API Key：CLI 参数 > 环境变量 > 配置文件。
pub fn resolve_api_key(provider: &str, cli_key: Option<&str>, cfg_key: Option<&str>) -> Option<String> {
    if let Some(k) = cli_key {
        return Some(k.to_string());
    }
    if let Ok(k) = std::env::var("NEKOSUB_API_KEY") {
        if !k.is_empty() {
            return Some(k);
        }
    }
    if let Some(p) = preset(provider) {
        for var in p.key_env {
            if let Ok(k) = std::env::var(var) {
                if !k.is_empty() {
                    return Some(k);
                }
            }
        }
    }
    cfg_key.filter(|k| !k.is_empty()).map(str::to_string)
}

pub fn lang_name(code: &str) -> &str {
    match code.to_ascii_lowercase().as_str() {
        "ja" => "日语",
        "zh" | "zh-cn" | "zh-tw" | "zh-hans" | "zh-hant" => "中文",
        "en" => "英语",
        "ko" => "韩语",
        "fr" => "法语",
        "de" => "德语",
        "es" => "西班牙语",
        "ru" => "俄语",
        _ => code,
    }
}

fn build_system_prompt(from: &str, to: &str, custom: Option<&str>) -> String {
    let from_name = lang_name(from);
    let to_name = lang_name(to);
    match custom {
        Some(t) => t.replace("{from}", from_name).replace("{to}", to_name),
        None => format!("将以下{from_name}翻译成{to_name}，只输出译文。"),
    }
}

fn build_user_content(text: &str, history: &[Turn]) -> String {
    if history.is_empty() {
        return text.to_string();
    }
    let mut out = String::new();
    for (i, turn) in history.iter().enumerate() {
        out.push_str(&format!(
            "（上文{}）原文：{}\n（上文{}）译文：{}\n",
            i + 1,
            turn.source,
            i + 1,
            turn.target
        ));
    }
    out.push_str(&format!("原文：{text}"));
    out
}

/// 任意 OpenAI / Anthropic 兼容端点的通用翻译器。
pub struct Translator {
    client: reqwest::Client,
    protocol: Protocol,
    base_url: String,
    api_key: String,
    model: String,
    max_tokens: u32,
    temperature: f32,
    system_prompt: String,
}

pub struct TranslatorOptions<'a> {
    pub provider: &'a str,
    pub protocol: &'a str,
    pub api_key: Option<String>,
    pub base_url: Option<&'a str>,
    pub model: Option<&'a str>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub prompt: Option<&'a str>,
    pub from: &'a str,
    pub to: &'a str,
}

impl Translator {
    pub fn new(opts: TranslatorOptions<'_>) -> Result<Self> {
        let p = preset(opts.provider).ok_or_else(|| {
            anyhow!(
                "未知翻译后端 \"{}\"，可选：{}",
                opts.provider,
                providers().join(" / ")
            )
        })?;

        let base_url = opts
            .base_url
            .filter(|s| !s.is_empty())
            .unwrap_or(p.base_url)
            .trim_end_matches('/')
            .to_string();
        if base_url.is_empty() {
            bail!("provider=generic 需要通过 --api-base 或配置 translate.base_url 指定 API 地址");
        }

        let model = opts
            .model
            .filter(|s| !s.is_empty())
            .unwrap_or(p.model)
            .to_string();
        if model.is_empty() {
            bail!(
                "后端 \"{}\" 需要通过 --model 或配置 translate.model 指定模型名",
                opts.provider
            );
        }

        let api_key = opts.api_key.unwrap_or_default();
        if api_key.is_empty() && !p.key_optional {
            bail!(
                "缺少 API Key：请设置环境变量 {}，或使用 --api-key / 配置 translate.api_key",
                p.key_env.join(" / ")
            );
        }

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        Ok(Self {
            client,
            protocol: Protocol::parse(opts.protocol)?,
            base_url,
            api_key,
            model,
            max_tokens: opts.max_tokens,
            temperature: opts.temperature,
            system_prompt: build_system_prompt(opts.from, opts.to, opts.prompt),
        })
    }

    pub async fn translate(&self, text: &str, history: &[Turn]) -> Result<String> {
        let mut last_err = anyhow!("翻译失败");
        for attempt in 0..2 {
            if attempt > 0 {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }
            match self.try_translate(text, history).await {
                Ok(t) => return Ok(t),
                Err(e) => last_err = e,
            }
        }
        Err(last_err)
    }

    async fn try_translate(&self, text: &str, history: &[Turn]) -> Result<String> {
        match self.protocol {
            Protocol::OpenAi => self.try_openai(text, history).await,
            Protocol::Anthropic => self.try_anthropic(text, history).await,
        }
    }

    async fn try_openai(&self, text: &str, history: &[Turn]) -> Result<String> {
        let url = openai_url(&self.base_url);
        let body = ChatRequest {
            model: &self.model,
            messages: vec![
                Message {
                    role: "system",
                    content: self.system_prompt.clone(),
                },
                Message {
                    role: "user",
                    content: build_user_content(text, history),
                },
            ],
            max_tokens: self.max_tokens,
            temperature: self.temperature,
        };

        let mut req = self.client.post(&url).json(&body);
        if !self.api_key.is_empty() {
            req = req.bearer_auth(&self.api_key);
        }

        let raw = send(req).await?;
        let parsed: ChatResponse =
            serde_json::from_str(&raw).map_err(|e| anyhow!("解析翻译 API 响应失败：{e}\n{raw}"))?;
        let content = parsed
            .choices
            .first()
            .and_then(|c| c.message.content.as_deref())
            .unwrap_or("");
        finish_text(content)
    }

    async fn try_anthropic(&self, text: &str, history: &[Turn]) -> Result<String> {
        let url = anthropic_url(&self.base_url);
        let body = AnthropicRequest {
            model: &self.model,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            system: self.system_prompt.clone(),
            messages: vec![Message {
                role: "user",
                content: build_user_content(text, history),
            }],
        };

        let req = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&body);

        let raw = send(req).await?;
        let parsed: AnthropicResponse =
            serde_json::from_str(&raw).map_err(|e| anyhow!("解析翻译 API 响应失败：{e}\n{raw}"))?;
        let content = parsed
            .content
            .iter()
            .filter_map(|b| b.text.as_deref())
            .collect::<Vec<_>>()
            .join("");
        finish_text(&content)
    }
}

fn openai_url(base: &str) -> String {
    format!("{}/chat/completions", base.trim_end_matches('/'))
}

fn anthropic_url(base: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.ends_with("/v1") {
        format!("{base}/messages")
    } else {
        format!("{base}/v1/messages")
    }
}

async fn send(req: reqwest::RequestBuilder) -> Result<String> {
    let resp = req.send().await.map_err(|e| anyhow!("请求翻译 API 失败：{e}"))?;
    let status = resp.status();
    let raw = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        let snippet: String = raw.chars().take(200).collect();
        bail!("翻译 API 返回 {status}：{snippet}");
    }
    Ok(raw)
}

fn finish_text(content: &str) -> Result<String> {
    let content = content.trim();
    if content.is_empty() {
        bail!("翻译 API 返回了空译文");
    }
    Ok(content.to_string())
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<Message>,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Serialize)]
struct AnthropicRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    temperature: f32,
    system: String,
    messages: Vec<Message>,
}

#[derive(Serialize)]
struct Message {
    role: &'static str,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: RespMessage,
}

#[derive(Deserialize)]
struct RespMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<ContentBlock>,
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(default)]
    text: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_resolve() {
        assert_eq!(preset("DeepSeek").unwrap().base_url, "https://api.deepseek.com/v1");
        assert_eq!(preset("qwen").unwrap().model, "qwen-plus");
        assert_eq!(
            preset("mimo").unwrap().base_url,
            "https://token-plan-cn.xiaomimimo.com/v1"
        );
        assert!(preset("nope").is_none());
    }

    #[test]
    fn protocol_parse() {
        assert_eq!(Protocol::parse("openai").unwrap(), Protocol::OpenAi);
        assert_eq!(Protocol::parse("Anthropic").unwrap(), Protocol::Anthropic);
        assert!(Protocol::parse("grpc").is_err());
    }

    #[test]
    fn endpoint_urls() {
        assert_eq!(
            openai_url("https://x.example/v1/"),
            "https://x.example/v1/chat/completions"
        );
        assert_eq!(
            anthropic_url("https://token-plan-cn.xiaomimimo.com/anthropic"),
            "https://token-plan-cn.xiaomimimo.com/anthropic/v1/messages"
        );
        assert_eq!(anthropic_url("https://y.example/anthropic/v1"), "https://y.example/anthropic/v1/messages");
    }

    #[test]
    fn anthropic_response_parse() {
        let raw = r#"{"content":[{"type":"text","text":"你好"},{"type":"text","text":"世界"}]}"#;
        let parsed: AnthropicResponse = serde_json::from_str(raw).unwrap();
        let text = parsed
            .content
            .iter()
            .filter_map(|b| b.text.as_deref())
            .collect::<Vec<_>>()
            .join("");
        assert_eq!(text, "你好世界");
    }

    #[test]
    fn system_prompt_placeholders() {
        let p = build_system_prompt("ja", "zh", None);
        assert!(p.contains("日语") && p.contains("中文"));
        let p = build_system_prompt("en", "zh", Some("translate {from}->{to}"));
        assert_eq!(p, "translate 英语->中文");
    }

    #[test]
    fn user_content_with_history() {
        let history = vec![Turn {
            source: "こんにちは".into(),
            target: "你好".into(),
        }];
        let s = build_user_content("ありがとう", &history);
        assert!(s.contains("上文1"));
        assert!(s.ends_with("原文：ありがとう"));
    }
}
