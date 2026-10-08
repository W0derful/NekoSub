use crate::asr::{self, AsrEvent};
use crate::audio;
use crate::config::Config;
use crate::display::{self, Renderer};
use crate::translate::{Turn, Translator, TranslatorOptions};
use anyhow::{Context, Result};
use cpal::traits::StreamTrait;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use yamabiko_whisper::AudioInputConfig;

/// 翻译结果（按序号重排后打印，保证字幕顺序与语音一致）。
type Translated = (u64, f64, String, Result<String>);

struct State {
    seq: u64,
    next_seq: u64,
    pending: BTreeMap<u64, (f64, String, Result<String>)>,
    history: VecDeque<Turn>,
    max_context: usize,
    /// 当前句子已确认的稳定词，VAD 确认句尾后才整句翻译
    buffer: String,
    buffer_start: f64,
}

impl State {
    fn new(max_context: usize) -> Self {
        Self {
            seq: 0,
            next_seq: 1,
            pending: BTreeMap::new(),
            history: VecDeque::new(),
            max_context,
            buffer: String::new(),
            buffer_start: 0.0,
        }
    }

    /// 按序打印已完成的翻译，保持输出顺序。
    fn flush(&mut self, renderer: &mut Renderer) {
        while let Some((ts, source, result)) = self.pending.remove(&self.next_seq) {
            match result {
                Ok(target) => {
                    renderer.print_turn(ts, &source, &target);
                    self.history.push_back(Turn {
                        source,
                        target,
                    });
                    while self.history.len() > self.max_context {
                        self.history.pop_front();
                    }
                }
                Err(e) => renderer.print_error(ts, &source, &e.to_string()),
            }
            self.next_seq += 1;
        }
    }
}

pub async fn run(cfg: &Config, cli_api_key: Option<&str>) -> Result<()> {
    // 1. 音频捕获（回调线程只做下混 + 投递）
    let (audio_tx, audio_rx) = std::sync::mpsc::channel::<Vec<f32>>();
    let mut capture = Some(audio::open_capture(
        cfg.audio.host.as_deref(),
        cfg.audio.device.as_deref(),
        audio_tx,
    )?);

    // 2. 加载模型并创建识别处理器（内置 Silero VAD 切句）
    let (asr_model, vad_model) = asr::load_models(&cfg.asr, &cfg.vad)?;
    let processor = asr::create_processor(&asr_model, &vad_model, &cfg.asr)?;
    let sep = processor.sep().to_string();

    // 3. ASR 工作线程 + 事件通道
    let input = AudioInputConfig::new(capture.as_ref().unwrap().sample_rate, 1)
        .with_process_interval_sec(cfg.asr.process_interval_sec);
    let (ev_tx, mut ev_rx) = tokio::sync::mpsc::unbounded_channel::<AsrEvent>();
    let worker = asr::spawn_worker(
        processor,
        input,
        audio_rx,
        ev_tx,
        cfg.asr.max_sentence_sec,
        cfg.vad.silence_reset_sec,
    )?;
    capture
        .as_ref()
        .unwrap()
        .stream
        .play()
        .context("启动音频流失败")?;

    // 4. 翻译与显示（翻译在独立任务中进行，不阻塞识别）
    let translator = Arc::new(Translator::new(TranslatorOptions {
        provider: &cfg.translate.provider,
        protocol: &cfg.translate.protocol,
        api_key: crate::translate::resolve_api_key(
            &cfg.translate.provider,
            cli_api_key,
            cfg.translate.api_key.as_deref(),
        ),
        base_url: cfg.translate.base_url.as_deref(),
        model: cfg.translate.model.as_deref(),
        max_tokens: cfg.translate.max_tokens,
        temperature: cfg.translate.temperature,
        prompt: cfg.translate.prompt.as_deref(),
        from: &cfg.asr.language,
        to: &cfg.translate.to,
    })?);

    let mut renderer = Renderer::new(
        cfg.display.timestamps,
        cfg.display.tentative,
        cfg.display
            .save_dir
            .as_deref()
            .map(crate::config::expand_tilde),
    );
    // context_sentences = 0 表示不带翻译上下文
    let mut state = State::new(cfg.translate.context_sentences);
    let (res_tx, mut res_rx) = tokio::sync::mpsc::unbounded_channel::<Translated>();
    let started = std::time::Instant::now();

    eprintln!(
        "NekoSub 就绪：{} → {}，翻译后端 {}（Ctrl-C 停止）",
        crate::translate::lang_name(&cfg.asr.language),
        crate::translate::lang_name(&cfg.translate.to),
        cfg.translate.provider
    );

    // 5. 主循环：识别事件 / 翻译结果 / Ctrl-C
    let mut stopping = false;
    loop {
        tokio::select! {
            ev = ev_rx.recv() => {
                let (out, finished) = match ev {
                    Some(AsrEvent::Process(out)) => (out, false),
                    Some(AsrEvent::Finish(out)) => (out, true),
                    None => break,
                };
                handle_output(out, finished, &sep, &mut renderer, &mut state, &translator, &res_tx, &started);
                state.flush(&mut renderer);
                if finished {
                    break;
                }
            }
            Some(msg) = res_rx.recv() => {
                let (seq, ts, source, result) = msg;
                state.pending.insert(seq, (ts, source, result));
                state.flush(&mut renderer);
            }
            _ = tokio::signal::ctrl_c(), if !stopping => {
                eprintln!("\n收到 Ctrl-C，正在停止…");
                stopping = true;
                // 关闭音频流，让 ASR 线程冲刷并退出
                drop(capture.take());
            }
        }
    }

    // 6. 等待在途翻译完成后按序输出
    while state.next_seq <= state.seq {
        match tokio::time::timeout(Duration::from_secs(20), res_rx.recv()).await {
            Ok(Some(msg)) => {
                let (seq, ts, source, result) = msg;
                state.pending.insert(seq, (ts, source, result));
                state.flush(&mut renderer);
            }
            _ => break,
        }
    }
    renderer.clear_tentative();

    let _ = worker.join().map_err(|_| anyhow::anyhow!("ASR 线程异常退出"));
    Ok(())
}

fn handle_output(
    out: yamabiko_whisper::ProcessOutput,
    force_finalize: bool,
    sep: &str,
    renderer: &mut Renderer,
    state: &mut State,
    translator: &Arc<Translator>,
    res_tx: &tokio::sync::mpsc::UnboundedSender<Translated>,
    started: &std::time::Instant,
) {
    // LocalAgreement-2 会逐批提交稳定词，先累积成当前句子
    if !out.committed.is_empty() {
        let piece = display::join_words(&out.committed, sep);
        // 过滤被拆成多个词的非语音标签短语（如 "[Speaking in Japanese]"）
        if !piece.is_empty() && !display::is_non_speech_tag(&piece) {
            if state.buffer.is_empty() {
                // 词时间戳是话语内相对时间，不可靠；用会话时钟
                state.buffer_start = started.elapsed().as_secs_f64();
            } else {
                state.buffer.push_str(sep);
            }
            state.buffer.push_str(&piece);
        }
    }

    // 只在 VAD 确认句尾（或流结束）时整句翻译，避免逐批重复调 API
    if (force_finalize || out.finalized_by_vad) && !state.buffer.is_empty() {
        let source = std::mem::take(&mut state.buffer);
        let ts = state.buffer_start;
        state.seq += 1;
        let seq = state.seq;
        let history: Vec<Turn> = state.history.iter().cloned().collect();
        let translator = translator.clone();
        let res_tx = res_tx.clone();
        tokio::spawn(async move {
            let result = translator.translate(&source, &history).await;
            let _ = res_tx.send((seq, ts, source, result));
        });
    }

    // 临时行：显示当前未确认句子（已确认部分 + 临时假设）
    let mut line = if state.buffer.is_empty() {
        display::join_words(&out.tentative, sep)
    } else if out.tentative.is_empty() {
        state.buffer.clone()
    } else {
        format!(
            "{}{}{}",
            state.buffer,
            sep,
            display::join_words(&out.tentative, sep)
        )
    };
    if display::is_non_speech_tag(&line) {
        line.clear();
    }
    renderer.render_tentative(&line);
}
