# score-rec

从混音里拆出各乐器，再识别鼓的击打和有音高乐器的音符，导出时间轴、MIDI 和 MusicXML。

当前识别的对象是人声、贝斯、吉他、钢琴和鼓组。鼓组再拆成底鼓、军鼓、通鼓、踩镲、叮叮镲、碎音镲。踩镲会再区分为开镲和闭镲。`other` 轨只保留音频，不识谱。

## 流程

1. 用 `htdemucs_ft` 从混音分离人声、鼓、贝斯。
2. 用 `htdemucs_6s` 分离吉他、钢琴，以及其余声音 `other`。
3. 用鼓组模型把鼓轨拆成六件。
4. 鼓件上检测击打。人声和贝斯做单音跟踪，吉他和钢琴用 Basic Pitch 做复音识别，再用频谱删掉没有能量支撑的音和弱谐波。
5. 写出时间轴、分轨 MIDI、总谱 MIDI 和五线谱。

乐谱按固定 120 BPM、4/4 量化到三十二分音符。时间是相对音频开头的秒数，程序不估计原曲速度和拍号。

输入若不是 16-bit PCM WAV，会先转成 PCM WAV 再送进分离模型。MP3 的编码不能直接拿来写 WAV。

分离在 CPU 上跑。`htdemucs_ft` 含四个网络，每个网络再做 2 次 shift，所以一首几分钟的歌通常要十几分钟。

## 安装

需要 Python 3.10 或更高版本。

```text
pip install -e .
```

吉他和钢琴依赖 `basic-pitch`。Python 3.11 及以上时，它声明的旧版 TensorFlow 往往装不上。本机推理用的是包内 ONNX 权重，可以这样装：

```text
pip install basic-pitch==0.4.0 --no-deps
```

`onnxruntime`、`pretty-midi`、`mir-eval`、`resampy` 已列在项目依赖里。没有系统 ffmpeg 时，程序会把 `imageio-ffmpeg` 自带的可执行文件拷到 `.score-rec/bin`。

模型在第一次运行时下载到 `.score-rec/models`。鼓组权重的原发布地址已经失效，需要按下面的地址自行放到该目录，文件名必须是 `MDX23C-DrumSep-aufr33-jarredou.ckpt`。

## 使用

```text
score-rec 歌曲.mp3
score-rec 歌曲.wav -o 输出目录
score-rec 歌曲.mp3 --no-stems
```

默认输出目录是 `score_rec_out/<文件名>/`。

| 文件 | 内容 |
|---|---|
| `timeline.txt` | 击打和音符的文本时间轴 |
| `timeline.json` | 同上，带 MIDI 音符号 |
| `drums.mid` | 鼓组 |
| `vocals.mid` `bass.mid` `guitar.mid` `piano.mid` | 有音符时才生成 |
| `score.mid` | 以上音轨合在一起 |
| `score.musicxml` | 分声部五线谱。钢琴用大谱表，鼓用打击乐谱号 |
| `stems/` | 六条混音分轨，以及六件鼓。`--no-stems` 时不保留 |

`stems/` 里各文件的含义：

| 文件 | 内容 |
|---|---|
| `vocals.wav` | 人声 |
| `drums.wav` | 整套鼓 |
| `bass.wav` | 贝斯 |
| `guitar.wav` | 吉他 |
| `piano.wav` | 钢琴 |
| `other.wav` | 没有分走的剩余声音 |
| `kick.wav` | 底鼓 |
| `snare.wav` | 军鼓 |
| `toms.wav` | 通鼓 |
| `hh.wav` | 踩镲，开镲和闭镲还在同一条里 |
| `ride.wav` | 叮叮镲 |
| `crash.wav` | 碎音镲 |

## 外部模型

### htdemucs_ft

Meta 的 Hybrid Transformer Demucs 微调包，用来分离人声、鼓、贝斯和其他。四个权重各负责一路，由 yaml 按对角权重合成。

权重，来自 Meta 公开文件库：

- https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th
- https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/d12395a8-e57c48e6.th
- https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/92cfc3b6-ef3bcb9c.th
- https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/04573f0d-f3cf25b2.th

名单来自 UVR 工具链维护者 TRvlvr，内容是指向上述官方权重的配置：

- https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/htdemucs_ft.yaml

权重是 Demucs 作者发布的原件。yaml 是社区封装。

### htdemucs_6s

Meta 的六轨 Demucs，在人声、鼓、贝斯、其他之外再分出吉他和钢琴。

- 权重：https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th
- 名单：https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/htdemucs_6s.yaml

人声、鼓、贝斯有公开分离指标。吉他和钢琴是后加的音源，Meta 说明过钢琴轨当时效果不好。吉他轨里也常混入其他乐器。

### MDX23C DrumSep

aufr33 与 jarredou 的六轨鼓组模型，只处理已经分开的鼓轨，输出底鼓、军鼓、通鼓、踩镲、叮叮镲、碎音镲。这是 2024 年的初版，作者标明训练未完成，SDR 约 10.8。2025 年的五轨版本原始权重现已不可得，本项目没有使用那一版。

作者原来的发布页 `github.com/jarredou/models` 已不可用。`audio-separator` 登记的 TRvlvr 下载地址也会 404。当前可用、且与独立镜像校验和一致的副本是：

- 权重：https://huggingface.co/Sucial/MSST-WebUI/resolve/main/All_Models/multi_stem_models/aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt
- SHA-256：`d2a4aa53eb584d21eead358a4e66d1882ad182911be018f052b5da73be9096d0`

下载后放到 `.score-rec/models/MDX23C-DrumSep-aufr33-jarredou.ckpt`。

网络结构配置由 python-audio-separator 发布：

- https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/config_drumsep_mdx23c.yaml

程序在解析模型列表时还会读取 TRvlvr 的清单，它不是神经网络权重：

- https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json

### Basic Pitch

Spotify Research 在 ICASSP 2022 发布的复音转录模型，随 `basic-pitch` 0.4.0 安装。`pitch.py` 实际加载的是包内文件：

`basic_pitch/saved_models/icassp_2022/nmp.onnx`

同一目录的 `nmp.tflite` 不使用。这是官方发布包里的推理权重。吉他使用更灵敏的音头阈值，钢琴更保守。识别后再用恒定 Q 频谱核对音高，并去掉同时响起、明显更弱的高八度谐波。

## 代码

```
score_rec/
├── cli.py         读命令行参数并启动
├── __main__.py    python -m score_rec 的入口
├── __init__.py    版本号
├── pipeline.py    按顺序调用分轨、识别和导出
├── separate.py    调用分离模型，写出各轨 wav
├── drums.py       在单面鼓上找击打，区分开镲和闭镲
├── pitch.py       识别人声、贝斯、吉他、钢琴
├── export.py      写时间轴、MIDI 和 MusicXML
└── spec.py        模型文件名、音域、鼓的 MIDI 音符号
```

`cli` 调用 `pipeline`。`pipeline` 用 `separate` 拆音频，用 `drums` 和 `pitch` 识别，用 `export` 写结果。`spec` 只放参数。

## 已知限制

鼓件之间会串音，叮叮镲、碎音镲和通鼓容易把别的击打记进来。吉他和钢琴轨若混入原曲没有单独分轨的乐器，那些声音也会被记成音符。人声和贝斯只跟踪一条旋律。乐谱不估计原曲的速度、拍号和调号。
