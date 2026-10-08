use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use yamabiko_whisper::Word;

const DIM: &str = "\x1b[2m";
const WHITE: &str = "\x1b[37m";
const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const RESET: &str = "\x1b[0m";

/// 翻译记录落盘：按日期一个 txt，追加写入。
struct Transcript {
    dir: Option<PathBuf>,
    date: String,
    file: Option<std::fs::File>,
    header_written: bool,
}

impl Transcript {
    fn new(dir: Option<PathBuf>) -> Self {
        Self {
            dir,
            date: String::new(),
            file: None,
            header_written: false,
        }
    }

    fn ensure_file(&mut self) -> Option<&mut std::fs::File> {
        let dir = self.dir.as_ref()?;
        let today = chrono::Local::now().format("%Y-%m-%d").to_string();
        if self.date != today || self.file.is_none() {
            if let Err(e) = std::fs::create_dir_all(dir) {
                eprintln!("警告：无法创建翻译记录目录 {}（{e}），本次不保存", dir.display());
                self.dir = None;
                return None;
            }
            let path = dir.join(format!("{today}.txt"));
            match OpenOptions::new().create(true).append(true).open(&path) {
                Ok(f) => {
                    self.file = Some(f);
                    self.date = today;
                    self.header_written = false;
                }
                Err(e) => {
                    eprintln!("警告：无法写入翻译记录 {}（{e}），本次不保存", path.display());
                    self.dir = None;
                    return None;
                }
            }
        }
        self.file.as_mut()
    }

    fn append(&mut self, lines: &[String]) {
        if self.ensure_file().is_none() {
            return;
        }
        let write_header = !self.header_written;
        self.header_written = true;
        let Some(file) = self.file.as_mut() else {
            return;
        };
        if write_header {
            let _ = writeln!(
                file,
                "\n=== 会话 {} 开始 ===",
                chrono::Local::now().format("%Y-%m-%d %H:%M:%S")
            );
        }
        for line in lines {
            let _ = writeln!(file, "{line}");
        }
        let _ = file.flush();
    }
}

pub struct Renderer {
    pub timestamps: bool,
    pub tentative: bool,
    tentative_visible: bool,
    transcript: Transcript,
}

impl Renderer {
    pub fn new(timestamps: bool, tentative: bool, save_dir: Option<PathBuf>) -> Self {
        Self {
            timestamps,
            tentative,
            tentative_visible: false,
            transcript: Transcript::new(save_dir),
        }
    }

    /// 识别中的临时行：灰色单行原地刷新。
    pub fn render_tentative(&mut self, text: &str) {
        if !self.tentative {
            return;
        }
        if text.is_empty() {
            self.clear_tentative();
            return;
        }
        eprint!("\r\x1b[K{DIM}… {text}{RESET}");
        let _ = std::io::stderr().flush();
        self.tentative_visible = true;
    }

    pub fn clear_tentative(&mut self) {
        if self.tentative_visible {
            eprint!("\r\x1b[K");
            let _ = std::io::stderr().flush();
            self.tentative_visible = false;
        }
    }

    /// 确认输出：原文 + 译文对照，永久保留在终端滚动历史里，并落盘保存。
    pub fn print_turn(&mut self, start_sec: f64, source: &str, target: &str) {
        self.clear_tentative();
        let ts = if self.timestamps {
            format!("[{}] ", fmt_ts(start_sec))
        } else {
            String::new()
        };
        println!("{ts}{WHITE}原文：{source}{RESET}");
        println!("{ts}{GREEN}译文：{target}{RESET}");
        let _ = std::io::stdout().flush();
        self.transcript.append(&[
            format!("[{}] 原文：{source}", fmt_ts(start_sec)),
            format!("[{}] 译文：{target}", fmt_ts(start_sec)),
        ]);
    }

    pub fn print_error(&mut self, start_sec: f64, source: &str, err: &str) {
        self.clear_tentative();
        let ts = if self.timestamps {
            format!("[{}] ", fmt_ts(start_sec))
        } else {
            String::new()
        };
        println!("{ts}{WHITE}原文：{source}{RESET}");
        println!("{ts}{RED}（翻译失败：{err}）{RESET}");
        let _ = std::io::stdout().flush();
        self.transcript.append(&[
            format!("[{}] 原文：{source}", fmt_ts(start_sec)),
            format!("[{}] （翻译失败：{err}）", fmt_ts(start_sec)),
        ]);
    }
}

/// 秒 → HH:MM:SS
pub fn fmt_ts(sec: f64) -> String {
    let total = sec.max(0.0) as u64;
    format!("{:02}:{:02}:{:02}", total / 3600, (total % 3600) / 60, total % 60)
}

/// Whisper 会为非语音输出 `[MUSIC]` / `[FOREIGN]` / `[NOISE]` 等标签，
/// 这类词不进字幕、不发翻译。
pub fn is_non_speech_tag(text: &str) -> bool {
    let t = text.trim();
    let Some(inner) = t.strip_prefix('[').and_then(|s| s.strip_suffix(']')) else {
        return false;
    };
    !inner.is_empty()
        && inner
            .chars()
            .all(|c| c.is_ascii_alphabetic() || c == ' ' || c == '-')
}

pub fn join_words(words: &[Word], sep: &str) -> String {
    words
        .iter()
        .filter(|w| !is_non_speech_tag(&w.text))
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join(sep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamp_format() {
        assert_eq!(fmt_ts(0.0), "00:00:00");
        assert_eq!(fmt_ts(3723.4), "01:02:03");
    }

    #[test]
    fn transcript_appends_to_daily_file() {
        let dir = std::env::temp_dir().join(format!("nekosub-test-{}", std::process::id()));
        let mut t = Transcript::new(Some(dir.clone()));
        t.append(&["[00:00:01] 原文：こんにちは".into(), "[00:00:01] 译文：你好".into()]);
        let today = chrono::Local::now().format("%Y-%m-%d").to_string();
        let content = std::fs::read_to_string(dir.join(format!("{today}.txt"))).unwrap();
        assert!(content.contains("会话"));
        assert!(content.contains("原文：こんにちは"));
        assert!(content.contains("译文：你好"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn transcript_disabled_writes_nothing() {
        let mut t = Transcript::new(None);
        t.append(&["x".into()]);
        assert!(t.dir.is_none());
    }

    #[test]
    fn non_speech_tags_filtered() {
        assert!(is_non_speech_tag("[MUSIC]"));
        assert!(is_non_speech_tag("[FOREIGN]"));
        assert!(is_non_speech_tag("[Speaking in Japanese]"));
        assert!(!is_non_speech_tag("こんにちは"));
        assert!(!is_non_speech_tag("[1]"));
        let words = vec![
            Word { start: 0.0, end: 0.5, text: "[MUSIC]".into() },
            Word { start: 0.5, end: 1.0, text: "你好".into() },
        ];
        assert_eq!(join_words(&words, ""), "你好");
    }
}
