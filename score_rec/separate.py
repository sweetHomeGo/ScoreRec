import gc
import logging
import os
import shutil
from pathlib import Path

import librosa
import soundfile as sf

from score_rec.spec import DEMUCS, FT_MODEL, FT_STEMS, KIT, KIT_MODEL, KIT_PARAMS, SIX_MODEL, SIX_STEMS


def workspace_dir():
    path = Path(__file__).resolve().parents[1] / ".score-rec"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_ffmpeg():
    if shutil.which("ffmpeg"):
        return
    import imageio_ffmpeg

    src = Path(imageio_ffmpeg.get_ffmpeg_exe())
    dest_dir = workspace_dir() / "bin"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "ffmpeg.exe"
    if not dest.exists() or dest.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dest)
    os.environ["PATH"] = str(dest_dir) + os.pathsep + os.environ.get("PATH", "")


def find_stem(paths, names):
    wanted = {name.lower() for name in names}
    for raw in paths:
        path = Path(raw)
        stem = path.stem.lower()
        if stem in wanted:
            return path
        for part in stem.split("_(")[1:]:
            token = part.split(")_", 1)[0]
            if token in wanted:
                return path
    return None


def _locate(path, out_dir):
    found = Path(path)
    if found.is_absolute() and found.is_file():
        return found
    candidate = out_dir / found.name
    if candidate.is_file():
        return candidate
    if found.is_file():
        return found
    return candidate


def _pcm_wav(audio, dest_dir):
    audio = Path(audio)
    if audio.suffix.lower() == ".wav" and str(sf.info(audio).subtype).startswith("PCM"):
        return audio
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "_pcm.wav"
    y, sr = librosa.load(str(audio), sr=None, mono=False)
    data = y.T if getattr(y, "ndim", 1) == 2 else y
    sf.write(dest, data, int(sr), subtype="PCM_16")
    return dest


def separate(audio, out_dir, model_dir, model, output_names, **params):
    from audio_separator.separator import Separator

    audio = _pcm_wav(audio, out_dir)
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
        return [_locate(path, out_dir) for path in separator.separate(str(audio), output_names)]
    finally:
        del separator
        gc.collect()


def save_stem(files, name, stem_dir, aliases=None):
    found = find_stem(files, aliases or (name,))
    if found is None or not found.is_file():
        raise RuntimeError(f"没有得到{name}。实际文件: {[str(path) for path in files]}")
    target = stem_dir / f"{name}.wav"
    if found.resolve() != target.resolve():
        if target.exists():
            target.unlink()
        shutil.move(str(found), target)
    return target


def split_mix(audio, stem_dir, model_dir):
    scratch = stem_dir.parent / "_work"
    print("1/4 分离人声、鼓、贝斯")
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        ft_files = separate(audio, scratch, model_dir, FT_MODEL, {name: name for name in FT_STEMS}, demucs_params=DEMUCS)
        stems = {name: save_stem(ft_files, name, stem_dir) for name in FT_STEMS}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    print("2/4 分离吉他、钢琴")
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        six_files = separate(audio, scratch, model_dir, SIX_MODEL, {name: name for name in SIX_STEMS}, demucs_params=DEMUCS)
        for name in SIX_STEMS:
            stems[name] = save_stem(six_files, name, stem_dir)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return stems


def split_kit(drums, stem_dir, model_dir):
    parts = separate(drums, stem_dir, model_dir, KIT_MODEL, {name: name for name in KIT}, mdxc_params=KIT_PARAMS)
    return {name: save_stem(parts, name, stem_dir, aliases) for name, aliases in KIT.items()}
