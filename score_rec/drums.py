import json
from dataclasses import dataclass
from pathlib import Path

import librosa
import mido
import numpy as np
from librosa.util import peak_pick

DRUMS = {
    "kick": {"name": "底鼓", "midi": 36, "gap": 0.06, "dur": 0.10},
    "snare": {"name": "军鼓", "midi": 38, "gap": 0.05, "dur": 0.10},
    "hh_closed": {"name": "闭镲", "midi": 42, "gap": 0.04, "dur": 0.06},
    "hh_open": {"name": "开镲", "midi": 46, "gap": 0.04, "dur": 0.28},
    "toms": {"name": "通鼓", "midi": 47, "gap": 0.05, "dur": 0.16},
    "ride": {"name": "叮叮镲", "midi": 51, "gap": 0.05, "dur": 0.18},
    "crash": {"name": "碎音镲", "midi": 49, "gap": 0.12, "dur": 0.45},
}
ORDER = ("kick", "snare", "hh_closed", "hh_open", "toms", "ride", "crash")


@dataclass(frozen=True)
class Hit:
    time: float
    drum: str
    velocity: int


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


def detect_onsets(y, sr, gap):
    y = np.asarray(y, dtype=np.float32)
    if y.size < sr // 10 or float(np.max(np.abs(y))) < 1e-5:
        return []
    hop = 512
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    top = float(np.max(env))
    if top < 1e-6:
        return []
    env_n = env / top
    wait = max(1, int(gap * sr / hop))
    frames = peak_pick(env_n, pre_max=3, post_max=3, pre_avg=8, post_avg=8, delta=0.15, wait=wait)
    peak = float(np.max(np.abs(y)))
    hits = []
    for frame in np.atleast_1d(frames):
        t = float(librosa.frames_to_time(int(frame), sr=sr, hop_length=hop))
        i = int(t * sr)
        window = y[max(0, i - hop) : i + hop * 2]
        if window.size == 0 or float(np.max(np.abs(window))) < peak * 0.07:
            continue
        velocity = int(np.clip(round(28 + 99 * float(env_n[int(frame)])), 1, 127))
        hits.append((t, velocity))
    return hits


def is_open_hat(y, sr, t, next_t):
    end = t + 0.30
    if next_t is not None:
        end = min(end, next_t - 0.005)
    if end - t < 0.10:
        return False
    seg = np.asarray(y[int(t * sr) : int(end * sr)], dtype=np.float32)
    win = max(1, int(0.02 * sr))
    n = seg.size // win
    if n < 3:
        return False
    rms = np.sqrt(np.mean(seg[: n * win].reshape(n, win) ** 2, axis=1))
    peak = float(np.max(rms[: max(1, int(0.04 * sr / win))]))
    if peak < 1e-6:
        return False
    quiet_at = next(
        (index for index in range(2, n - 1) if rms[index] < peak * 0.25 and rms[index + 1] < peak * 0.25),
        None,
    )
    decay = (n * win / sr) if quiet_at is None else (quiet_at * win / sr)
    return decay >= 0.12


def transcribe_stems(stem_paths, sr=44100):
    hits = []
    for drum, path in stem_paths.items():
        y, file_sr = librosa.load(path, sr=sr, mono=True)
        gap = 0.04 if drum == "hh" else DRUMS[drum]["gap"]
        found = detect_onsets(y, file_sr, gap)
        for index, (t, velocity) in enumerate(found):
            kind = drum
            if drum == "hh":
                next_t = found[index + 1][0] if index + 1 < len(found) else None
                kind = "hh_open" if is_open_hat(y, file_sr, t, next_t) else "hh_closed"
            hits.append(Hit(round(t, 3), kind, velocity))
    hits.sort(key=lambda hit: (hit.time, hit.drum))
    return hits


def _pad(text, width):
    used = sum(2 if ord(ch) > 127 else 1 for ch in text)
    return text + " " * max(0, width - used)


def format_timeline(hits):
    counts = {}
    for hit in hits:
        counts[hit.drum] = counts.get(hit.drum, 0) + 1
    head = "  ".join(f"{DRUMS[drum]['name']} {counts.get(drum, 0)}" for drum in ORDER)
    lines = [head, ""]
    for hit in hits:
        lines.append(f"{hit.time:8.3f}  {_pad(DRUMS[hit.drum]['name'], 8)}{hit.velocity:3d}")
    return "\n".join(lines) + "\n"


def write_timeline(hits, source, out_dir):
    out_dir = Path(out_dir)
    text_path = out_dir / "timeline.txt"
    json_path = out_dir / "timeline.json"
    text_path.write_text(format_timeline(hits), encoding="utf-8")
    payload = {
        "source": str(source),
        "hits": [
            {
                "time": hit.time,
                "drum": hit.drum,
                "name": DRUMS[hit.drum]["name"],
                "velocity": hit.velocity,
                "midi": DRUMS[hit.drum]["midi"],
            }
            for hit in hits
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return text_path, json_path


def write_midi(hits, path, bpm=120):
    tpb = 480
    mid = mido.MidiFile(ticks_per_beat=tpb)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    scale = bpm * tpb / 60
    events = []
    for hit in hits:
        spec = DRUMS[hit.drum]
        start = int(round(hit.time * scale))
        end = max(start + 1, int(round((hit.time + spec["dur"]) * scale)))
        events.append((start, 1, spec["midi"], hit.velocity))
        events.append((end, 0, spec["midi"], 0))
    events.sort()
    last = 0
    for tick, on, note, velocity in events:
        kind = "note_on" if on else "note_off"
        track.append(mido.Message(kind, channel=9, note=note, velocity=velocity, time=tick - last))
        last = tick
    mid.save(path)
    return Path(path)
