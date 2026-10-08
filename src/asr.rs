use crate::config::{AsrConfig, VadConfig};
use anyhow::{Context, Result, bail};
use std::sync::mpsc::Receiver;
use std::thread::JoinHandle;
use tokio::sync::mpsc::UnboundedSender;
use yamabiko_whisper::{
    AudioInputConfig, BackendConfig, OnlineAsrModel, OnlineAsrProcessor, ProcessOutput, SAMPLE_RATE,
    VadModel,
};

pub enum AsrEvent {
    Process(ProcessOutput),
    Finish(ProcessOutput),
}

fn check_model(path: &str, what: &str) -> Result<std::path::PathBuf> {
    let p = crate::config::expand_tilde(path);
    if !p.exists() {
        bail!(
            "{what}不存在：{}\n  Whisper 模型：https://huggingface.co/ggerganov/whisper.cpp（如 ggml-medium.bin）\n  VAD 模型：https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin",
            p.display()
        );
    }
    Ok(p)
}

/// 加载 Whisper 与 Silero VAD 模型（较慢，启动时执行一次）。
pub fn load_models(cfg: &AsrConfig, vad: &VadConfig) -> Result<(OnlineAsrModel, VadModel)> {
    let model_path = check_model(&cfg.model, "Whisper 模型")?;
    let vad_path = check_model(&cfg.vad_model, "VAD 模型")?;

    let backend = if cfg.cpu {
        BackendConfig::cpu()
    } else {
        let mut b = BackendConfig::default();
        if let Some(d) = cfg.gpu_device {
            b = b.with_gpu_device(d);
        }
        b
    };

    eprintln!("加载 Whisper 模型：{}", model_path.display());
    let model =
        OnlineAsrModel::load_with_backend(&model_path, backend).context("加载 Whisper 模型失败")?;

    eprintln!("加载 VAD 模型：{}", vad_path.display());
    let mut vad_cfg = yamabiko_whisper::VadConfig::default();
    vad_cfg.threshold = vad.threshold;
    vad_cfg.min_speech_ms = vad.min_speech_ms;
    vad_cfg.min_silence_ms = vad.min_silence_ms;
    vad_cfg.silence_reset_sec = vad.silence_reset_sec;
    let vad_model = VadModel::load_with_config(&vad_path, vad_cfg).context("加载 VAD 模型失败")?;

    Ok((model, vad_model))
}

pub fn create_processor(
    model: &OnlineAsrModel,
    vad_model: &VadModel,
    cfg: &AsrConfig,
) -> Result<OnlineAsrProcessor> {
    let mut acfg = yamabiko_whisper::OnlineAsrConfig::new(&cfg.language);
    if let Some(t) = cfg.threads {
        acfg = acfg.with_n_threads(t as i32);
    }
    if cfg.no_context {
        // 不把先前的识别文本作为 Whisper 初始提示，每句独立解码
        acfg = acfg.with_prompt_char_budget(0);
    }
    model
        .create_processor_with_config_and_vad(acfg, vad_model)
        .context("创建识别处理器失败")
}

/// whisper.cpp 的 Silero VAD 对全零输入会给出不稳定概率（阈值拉满也放行），
/// 导致静音段被送进 Whisper 产生幻觉。给纯数字静音注入极小抖动，
/// 让 VAD 正常判为非语音，静音计数/句尾 flush 逻辑保持不变。
fn dither_if_silent(chunk: &mut [f32]) {
    if chunk.iter().all(|&s| s == 0.0) {
        for (i, s) in chunk.iter_mut().enumerate() {
            *s = if i % 2 == 0 { 1e-5 } else { -1e-5 };
        }
    }
}

/// ASR 工作线程：消费音频 chunk，产出识别事件。Whisper 推理是阻塞的，
/// 必须独立于音频回调与异步任务运行。
///
/// 连续说话超过 `max_sentence_sec` 仍未出现句尾静音时，注入一段合成静音，
/// 触发管道内部的 VAD 复位（冲刷当前句并重置解码状态）——限制句子长度的
/// 同时打断 whisper 长缓冲下的复读循环。
pub fn spawn_worker(
    processor: OnlineAsrProcessor,
    input: AudioInputConfig,
    audio_rx: Receiver<Vec<f32>>,
    ev_tx: UnboundedSender<AsrEvent>,
    max_sentence_sec: f64,
    silence_reset_sec: f64,
) -> Result<JoinHandle<Result<()>>> {
    let handle = std::thread::Builder::new()
        .name("nekosub-asr".to_string())
        .spawn(move || -> Result<()> {
            let mut pipeline = yamabiko_whisper::AsrPipeline::new(processor, input)?;
            let mut sec_since_finalize = 0.0f64;
            while let Ok(mut chunk) = audio_rx.recv() {
                if chunk.is_empty() {
                    continue;
                }
                dither_if_silent(&mut chunk);
                sec_since_finalize += chunk.len() as f64 / SAMPLE_RATE as f64;
                if let Some(output) = pipeline.push_mono(&chunk)? {
                    if output.finalized_by_vad {
                        sec_since_finalize = 0.0;
                    }
                    if ev_tx.send(AsrEvent::Process(output)).is_err() {
                        return Ok(());
                    }
                }
                if max_sentence_sec > 0.0 && sec_since_finalize > max_sentence_sec {
                    let silence = synthetic_silence(silence_reset_sec + 0.5);
                    if let Some(output) = pipeline.push_mono(&silence)? {
                        if ev_tx.send(AsrEvent::Process(output)).is_err() {
                            return Ok(());
                        }
                    }
                    sec_since_finalize = 0.0;
                }
            }
            // 音频流关闭：冲刷管道里剩余的识别结果
            let final_output = pipeline.finish()?;
            let _ = ev_tx.send(AsrEvent::Finish(final_output));
            Ok(())
        })
        .context("创建 ASR 线程失败")?;
    Ok(handle)
}

/// 极小抖动的合成静音：精确全零会触发 VAD 对静音的误判（见 dither_if_silent）。
fn synthetic_silence(sec: f64) -> Vec<f32> {
    let n = (sec * SAMPLE_RATE as f64) as usize;
    (0..n)
        .map(|i| if i % 2 == 0 { 1e-5 } else { -1e-5 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dither_only_touches_digital_silence() {
        let mut silent = vec![0.0f32; 4];
        dither_if_silent(&mut silent);
        assert!(silent.iter().all(|&s| s != 0.0));

        let mut speech = vec![0.5f32, 0.0, -0.25, 0.0];
        dither_if_silent(&mut speech);
        assert_eq!(speech, vec![0.5, 0.0, -0.25, 0.0]);
    }

    #[test]
    fn synthetic_silence_is_not_all_zero() {
        let s = synthetic_silence(0.1);
        assert_eq!(s.len(), SAMPLE_RATE / 10);
        assert!(s.iter().all(|&v| v != 0.0));
    }
}
