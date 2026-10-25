import librosa
import numpy as np
from librosa.util import peak_pick

from score_rec.spec import DRUMS, Hit


def detect_onsets(y, sr, gap):
    y = np.asarray(y, dtype=np.float32)
    if y.size < sr // 10 or float(np.max(np.abs(y))) < 1e-5:
        return []
    hop = 512
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    scale = float(np.percentile(env, 95))
    if scale < 1e-6:
        return []
    env_n = env / scale
    wait = max(1, int(gap * sr / hop))
    frames = peak_pick(env_n, pre_max=3, post_max=3, pre_avg=8, post_avg=8, delta=0.1, wait=wait)
    peak = float(np.max(np.abs(y)))
    floor = peak * 0.02
    ref = max(float(np.percentile(np.abs(y), 99)), 1e-8)
    hits = []
    for frame in np.atleast_1d(frames):
        t = float(librosa.frames_to_time(int(frame), sr=sr, hop_length=hop))
        i = int(t * sr)
        window = y[max(0, i - hop) : i + hop * 2]
        if window.size == 0:
            continue
        local = float(np.max(np.abs(window)))
        if local < floor:
            continue
        velocity = int(np.clip(round(127 * min(1.0, local / ref) ** 0.5), 1, 127))
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
