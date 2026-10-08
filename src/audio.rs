use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{SampleFormat, SizedSample, StreamConfig};
use std::sync::mpsc::Sender;
use yamabiko_whisper::AudioSample;

pub struct Capture {
    pub stream: cpal::Stream,
    pub sample_rate: u32,
}

/// 选择音频后端：优先 PulseAudio（PipeWire 系统经 pipewire-pulse 兼容层同样可用，
/// 且能枚举 `.monitor` 系统音频源），其次 PipeWire，最后默认后端。
pub fn select_host(name: Option<&str>) -> Result<cpal::Host> {
    if let Some(n) = name {
        let id: cpal::HostId = n
            .parse()
            .with_context(|| format!("未知音频后端 \"{n}\"，可选：pulseaudio / pipewire / alsa"))?;
        return cpal::host_from_id(id).with_context(|| format!("无法初始化音频后端 {n}"));
    }
    for candidate in ["pulseaudio", "pipewire"] {
        if let Ok(id) = candidate.parse::<cpal::HostId>() {
            if let Ok(host) = cpal::host_from_id(id) {
                return Ok(host);
            }
        }
    }
    Ok(cpal::default_host())
}

/// 列出输入设备（id, 显示名）。
pub fn list_inputs(host_name: Option<&str>) -> Result<Vec<(String, String)>> {
    let host = select_host(host_name)?;
    let mut out = Vec::new();
    for dev in host.input_devices().context("枚举输入设备失败")? {
        let id = dev
            .id()
            .map(|i| i.id().to_string())
            .unwrap_or_else(|_| dev.to_string());
        out.push((id, dev.to_string()));
    }
    Ok(out)
}

pub fn print_devices(host_name: Option<&str>) -> Result<()> {
    let host = select_host(host_name)?;
    println!("音频后端：{}", host.id());
    let inputs = list_inputs(host_name)?;
    if inputs.is_empty() {
        println!("（没有发现输入设备）");
    }
    for (id, name) in inputs {
        println!("  {id}  ({name})");
    }
    Ok(())
}

fn default_monitor_name() -> Option<String> {
    let out = std::process::Command::new("pactl")
        .arg("get-default-sink")
        .output()
        .ok()?;
    let sink = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!sink.is_empty()).then(|| format!("{sink}.monitor"))
}

/// 自动选择系统音频 monitor 源；`query` 非空时按子串匹配设备 id 或名称。
fn pick_device(host: &cpal::Host, query: Option<&str>) -> Result<cpal::Device> {
    let devices: Vec<cpal::Device> = host
        .input_devices()
        .context("枚举输入设备失败")?
        .collect();
    if devices.is_empty() {
        bail!("没有可用的输入设备");
    }

    let id_of = |d: &cpal::Device| -> String {
        d.id().map(|i| i.id().to_string()).unwrap_or_default()
    };

    if let Some(q) = query {
        let q = q.to_lowercase();
        for d in &devices {
            if id_of(d).to_lowercase().contains(&q) || d.to_string().to_lowercase().contains(&q) {
                return Ok(d.clone());
            }
        }
        bail!("没有匹配 \"{q}\" 的输入设备，用 --list-devices 查看可用设备");
    }

    // 优先默认 sink 对应的 monitor，其次任意 monitor，最后退回默认输入设备
    if let Some(want) = default_monitor_name() {
        for d in &devices {
            if id_of(d) == want {
                return Ok(d.clone());
            }
        }
    }
    for d in &devices {
        if id_of(d).to_lowercase().contains(".monitor") {
            return Ok(d.clone());
        }
    }
    if let Some(d) = host.default_input_device() {
        eprintln!(
            "提示：未找到系统音频 monitor 源，改用默认输入设备 {}（通常是麦克风）",
            d.to_string()
        );
        return Ok(d);
    }
    bail!("找不到可用的输入设备，用 --list-devices 查看")
}

/// 打开系统音频捕获，音频回调只做下混并投递到 channel，不做任何计算。
pub fn open_capture(
    host_name: Option<&str>,
    device_query: Option<&str>,
    tx: Sender<Vec<f32>>,
) -> Result<Capture> {
    let host = select_host(host_name)?;
    let device = pick_device(&host, device_query)?;
    let label = format!("{} [{}]", device.to_string(), device.id().map(|i| i.id().to_string()).unwrap_or_default());
    let supported = device
        .default_input_config()
        .context("查询输入设备默认配置失败")?;
    let sample_rate = supported.sample_rate();
    let channels = supported.channels() as usize;
    let sample_format = supported.sample_format();
    let stream_config: StreamConfig = supported.into();

    eprintln!(
        "音频输入：{label}\n  {sample_rate} Hz / {channels} ch / {sample_format:?}（内部重采样为 16kHz 单声道）"
    );

    macro_rules! build {
        ($ty:ty) => {
            build_input_stream::<$ty>(&device, &stream_config, channels, tx)?
        };
    }
    let stream = match sample_format {
        SampleFormat::F32 => build!(f32),
        SampleFormat::F64 => build!(f64),
        SampleFormat::I8 => build!(i8),
        SampleFormat::I16 => build!(i16),
        SampleFormat::I32 => build!(i32),
        SampleFormat::I64 => build!(i64),
        SampleFormat::U8 => build!(u8),
        SampleFormat::U16 => build!(u16),
        SampleFormat::U32 => build!(u32),
        SampleFormat::U64 => build!(u64),
        other => bail!("不支持的采样格式：{other:?}"),
    };

    Ok(Capture {
        stream,
        sample_rate,
    })
}

fn build_input_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    channels: usize,
    tx: Sender<Vec<f32>>,
) -> Result<cpal::Stream>
where
    T: SizedSample + AudioSample + Send + 'static,
{
    let err_fn = |err| eprintln!("音频流出错：{err}");
    Ok(device.build_input_stream::<T, _, _>(
        config.clone(),
        move |data, _| {
            let _ = tx.send(yamabiko_whisper::downmix_interleaved(data, channels));
        },
        err_fn,
        None,
    )?)
}
