import argparse
import gc
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

from score_rec.drums import find_stem, transcribe_stems, write_midi, write_timeline

DRUM_MODEL = "htdemucs.yaml"
KIT_MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
KIT = {
    "kick": ("kick",),
    "snare": ("snare",),
    "toms": ("toms", "tom"),
    "hh": ("hh", "hihat", "hi-hat"),
    "ride": ("ride",),
    "crash": ("crash",),
}


def build_parser():
    parser = argparse.ArgumentParser(description="从混音中识别鼓，导出时间轴和 MIDI")
    parser.add_argument("audio", type=Path, help="音乐文件，例如 mp3、wav、flac")
    parser.add_argument("-o", "--output", type=Path, help="输出目录，默认 score_rec_out/<文件名>")
    parser.add_argument("--no-stems", action="store_true", help="不保留分离出的音频")
    return parser


def ensure_ffmpeg():
    if shutil.which("ffmpeg"):
        return
    import imageio_ffmpeg

    src = Path(imageio_ffmpeg.get_ffmpeg_exe())
    dest_dir = Path.home() / ".score-rec" / "bin"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "ffmpeg.exe"
    if not dest.exists() or dest.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dest)
    os.environ["PATH"] = str(dest_dir) + os.pathsep + os.environ.get("PATH", "")


def locate(path, out_dir):
    found = Path(path)
    if found.is_file():
        return found
    return out_dir / found


def separate(audio, out_dir, model_dir, model, output_names, **params):
    from audio_separator.separator import Separator

    separator = None
    try:
        separator = Separator(
            log_level=logging.INFO,
            model_file_dir=str(model_dir),
            output_dir=str(out_dir),
            output_format="WAV",
            use_soundfile=True,
            use_autocast=False,
            **params,
        )
        separator.load_model(model)
        return [locate(path, out_dir) for path in separator.separate(str(audio), output_names)]
    finally:
        del separator
        gc.collect()


def take(files, names, label):
    found = find_stem(files, names)
    if found is None or not found.is_file():
        raise RuntimeError(f"没有得到{label}。实际文件: {[str(path) for path in files]}")
    return found


def run(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    audio = args.audio.expanduser().resolve()
    if not audio.is_file():
        raise SystemExit(f"找不到音频文件: {audio}")
    ensure_ffmpeg()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("未找到 ffmpeg。")

    out_dir = (args.output or Path("score_rec_out") / audio.stem).resolve()
    stem_dir = out_dir / "stems"
    model_dir = Path.home() / ".score-rec" / "models"
    stem_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out_dir}")

    print("1/3 从混音中分离鼓轨")
    with tempfile.TemporaryDirectory(prefix="score-rec-") as temp:
        drums = take(
            separate(
                audio,
                Path(temp),
                model_dir,
                DRUM_MODEL,
                {"Drums": "drums"},
                output_single_stem="Drums",
                demucs_params={"segment_size": "Default", "shifts": 1, "overlap": 0.25, "segments_enabled": True},
            ),
            ("drums",),
            "鼓轨",
        )
        drum_wav = stem_dir / "drums.wav"
        shutil.copy2(drums, drum_wav)

    print("2/3 拆分底鼓、军鼓、踩镲、通鼓、叮叮镲、碎音镲")
    parts = separate(
        drum_wav,
        stem_dir,
        model_dir,
        KIT_MODEL,
        {name: name for name in KIT},
        mdxc_params={
            "segment_size": 256,
            "override_model_segment_size": False,
            "batch_size": 1,
            "overlap": 2,
            "pitch_shift": 0,
            "process_all_stems": True,
        },
    )
    stem_paths = {}
    for name, aliases in KIT.items():
        found = take(parts, aliases, name)
        target = stem_dir / f"{name}.wav"
        if found.resolve() != target.resolve():
            if target.exists():
                target.unlink()
            shutil.move(found, target)
        stem_paths[name] = target

    print("3/3 检测击打时间")
    hits = transcribe_stems(stem_paths)
    text_path, json_path = write_timeline(hits, audio, out_dir)
    midi_path = write_midi(hits, out_dir / "drums.mid")
    if args.no_stems:
        shutil.rmtree(stem_dir, ignore_errors=True)

    print(text_path.read_text(encoding="utf-8").splitlines()[0])
    print(f"时间轴: {text_path}")
    print(f"JSON: {json_path}")
    print(f"MIDI: {midi_path}")


def main(argv=None):
    try:
        run(build_parser().parse_args(argv))
    except BrokenPipeError:
        sys.exit(0)


if __name__ == "__main__":
    main()
