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

/// 上下文放 system 而不是 user：小翻译模型会把 user 消息整体当待译文本，
/// 上下文混进去会被原样复述出来。
fn append_context(mut prompt: String, history: &[Turn]) -> String {
    if history.is_empty() {
        return prompt;
    }
    prompt.push_str("\n\n对话上文（仅供理解语境，不要输出，只翻译用户消息）：");
    for (i, turn) in history.iter().enumerate() {
        prompt.push_str(&format!("\n{}. {} → {}", i + 1, turn.source, turn.target));
    }
    prompt
}

/// 输出疑似"复述/总结"而非单句译文：空、多行、句号过多、或长度远超原文。
fn is_suspicious(source: &str, out: &str) -> bool {
    let out = out.trim();
    if out.is_empty() {
        return true;
    }
    let lines = out.lines().filter(|l| !l.trim().is_empty()).count();
    if lines > 1 {
        return true;
    }
    if out.matches('。').count() > 2 {
        return true;
    }
    out.chars().count() > source.chars().count() * 4 + 30
}

/// 小翻译模型偶尔会把提示模板复述进译文（出现「原文：」「译文：」「上文」等标记）
/// 或整段总结上文，只保留真正的译文部分，防止污染上下文后滚雪球。
pub fn sanitize_translation(raw: &str) -> String {
    let mut s = raw.trim();
    loop {
        let cut = match (s.rfind("译文："), s.rfind("原文：")) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        let Some(i) = cut else { break };
        let marker_len = if s[i..].starts_with("译文：") {
            "译文：".len()
        } else {
            "原文：".len()
        };
        s = s[i + marker_len..].trim();
    }
    // 多行输出取最后一行：总结式复述里当前句的译文总是排在最后
    if s.contains('\n') || s.contains("（上文") {
        s = s.lines().map(str::trim).filter(|l| !l.is_empty()).last().unwrap_or("");
    }
    s.to_string()
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
                Ok(raw) => {
                    // 小翻译模型有时把上下文整段复述/总结出来，此时退回无上下文重译
                    if history.is_empty() || !is_suspicious(text, &raw) {
                        return finish_text(&raw);
                    }
                    match self.try_translate(text, &[]).await {
                        Ok(clean) => return finish_text(&clean),
                        Err(e) => last_err = e,
                    }
                }
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
                    content: append_context(self.system_prompt.clone(), history),
                },
                Message {
                    role: "user",
                    content: text.to_string(),
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
        Ok(content.trim().to_string())
    }

    async fn try_anthropic(&self, text: &str, history: &[Turn]) -> Result<String> {
        let url = anthropic_url(&self.base_url);
        let body = AnthropicRequest {
            model: &self.model,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            system: append_context(self.system_prompt.clone(), history),
            messages: vec![Message {
                role: "user",
                content: text.to_string(),
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
        Ok(content.trim().to_string())
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
    let content = sanitize_translation(content);
    if content.is_empty() {
        bail!("翻译 API 返回了空译文");
    }
    Ok(content)
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
    fn context_goes_to_system_not_user() {
        let history = vec![Turn {
            source: "こんにちは".into(),
            target: "你好".into(),
        }];
        let s = append_context("只输出译文。".to_string(), &history);
        assert!(s.contains("对话上文"));
        assert!(s.contains("こんにちは → 你好"));
        assert_eq!(append_context("x".into(), &[]), "x");
    }

    #[test]
    fn sanitize_strips_echoed_prompt() {
        // 实际日志里的复述形态：多层嵌套也要修干净
        assert_eq!(sanitize_translation("你好"), "你好");
        assert_eq!(
            sanitize_translation("（上文1）原文：これ?\n（上文1）译文：这个？"),
            "这个？"
        );
        assert_eq!(
            sanitize_translation("（上文1）原文：啊\n（上文1）译文：啊\n原文：死亡之旅1245日元"),
            "死亡之旅1245日元"
        );
        assert_eq!(
            sanitize_translation("（上文2）译文：（上文1）原文：えっとバーナー\n（上文1）译文：呃，巴纳"),
            "呃，巴纳"
        );
    }

    #[test]
    fn sanitize_drops_recap_blocks() {
        // 实际日志里的"总结式复述"：多行流水账，当前句译文在最后
        let recap = "报告游戏的话，虽然可以购买，不过价格只有450日元而已。\n是啊，今天真是令人惊讶啊。\nyoutube也……\n是的，两个人一起购买的话……";
        assert_eq!(sanitize_translation(recap), "是的，两个人一起购买的话……");
    }

    #[test]
    fn suspicious_outputs_detected() {
        let recap = "报告游戏的话，虽然可以购买。\n是啊，今天真是令人惊讶啊。\n是的，两个人一起购买的话……";
        assert!(is_suspicious("休みにだいぶ休みになってるやつ買っちゃう", recap));
        assert!(is_suspicious("ねえ", ""));
        assert!(is_suspicious("ねえ", "嗯。是啊。好的。知道了。"));
        assert!(!is_suspicious("そうねみんなのおすすめいっぱい聞けて面白かったな", "是啊，能听到很多人的推荐，真是有趣啊。"));
    }
}
