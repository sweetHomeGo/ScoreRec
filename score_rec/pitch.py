import contextlib
import io
import logging
from pathlib import Path

import librosa
import numpy as np
from librosa.util import peak_pick

from score_rec.spec import PARTS, Note

HOP = 512
FMIN_MIDI = 21
N_BINS = 88


def _velocity(level):
    return int(np.clip(round(127 * level), 1, 127))


def _tuning(y, sr):
    if y.size < sr // 2:
        return 0.0
    value = float(librosa.estimate_tuning(y=y, sr=sr, fmin=32.0, fmax=1200.0))
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, -0.5, 0.5))


def _attacks(y, sr, hop):
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    if env.size == 0 or float(np.max(env)) < 1e-6:
        return []
    wait = max(1, int(0.08 * sr / hop))
    frames = peak_pick(env / float(np.max(env)), pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.2, wait=wait)
    times = []
    for frame in np.atleast_1d(frames):
        t = float(librosa.frames_to_time(int(frame), sr=sr, hop_length=hop))
        pre = y[max(0, int((t - 0.05) * sr)) : max(1, int(t * sr))]
        post = y[int(t * sr) : min(y.size, int((t + 0.05) * sr))]
        pre_rms = float(np.sqrt(np.mean(pre * pre))) if pre.size else 0.0
        post_rms = float(np.sqrt(np.mean(post * post))) if post.size else 0.0
        if post_rms > pre_rms * 1.25 and post_rms > 0.04:
            times.append(t)
    return times


def _merge_overlaps(notes):
    ordered = sorted(notes, key=lambda note: (note.part, note.pitch, note.time))
    merged = []
    for note in ordered:
        if merged and merged[-1].part == note.part and merged[-1].pitch == note.pitch and note.time < merged[-1].end - 0.02:
            prev = merged[-1]
            merged[-1] = Note(prev.time, max(prev.end, note.end), prev.pitch, max(prev.velocity, note.velocity), prev.part)
        else:
            merged.append(note)
    merged.sort(key=lambda note: (note.time, note.part, note.pitch))
    return merged


def _bin(vec, midi, fmin_midi):
    index = int(midi) - fmin_midi
    if index < 0 or index >= len(vec):
        return 0.0
    return float(vec[index])


def align_pitch(midi, vec, fmin_midi=FMIN_MIDI):
    best = int(midi)
    best_energy = _bin(vec, best, fmin_midi)
    for candidate in (best - 1, best + 1):
        energy = _bin(vec, candidate, fmin_midi)
        if energy > best_energy * 1.25:
            best, best_energy = candidate, energy
    return best


def lower_octave(midi, vec, fmin_midi=FMIN_MIDI, lowest=0):
    low = int(midi) - 12
    if low < lowest:
        return int(midi)
    fundamental = _bin(vec, low, fmin_midi)
    overtone = _bin(vec, midi, fmin_midi)
    if fundamental > overtone * 1.15 and fundamental >= _bin(vec, low - 1, fmin_midi) and fundamental >= _bin(vec, low + 1, fmin_midi):
        return low
    return int(midi)


def _supported(midi, vec, fmin_midi):
    center = _bin(vec, midi, fmin_midi)
    if center <= 0:
        return False
    if center < max(_bin(vec, midi - 1, fmin_midi), _bin(vec, midi + 1, fmin_midi)) * 0.75:
        return False
    return center >= float(np.max(vec)) * 0.035


def suppress_harmonics(notes, curves, fmin_midi=FMIN_MIDI):
    dropped = set()
    order = sorted(range(len(notes)), key=lambda index: (notes[index].time, notes[index].pitch))
    for index in order:
        if index in dropped:
            continue
        for other in order:
            if other == index or other in dropped:
                continue
            low, high = notes[index], notes[other]
            if high.pitch - low.pitch not in (12, 19, 24):
                continue
            if abs(high.time - low.time) > 0.045:
                continue
            if high.end - high.time > max(0.35, (low.end - low.time) * 1.6):
                continue
            if _bin(curves[other], high.pitch, fmin_midi) < _bin(curves[index], low.pitch, fmin_midi) * 0.55:
                dropped.add(other)
    return [note for index, note in enumerate(notes) if index not in dropped]


def _snap_start(note, attacks):
    window = [time for time in attacks if note.time - 0.06 <= time <= note.time + 0.02 and time < note.end - 0.06]
    if not window:
        return note
    return Note(round(max(window), 3), note.end, note.pitch, note.velocity, note.part)


def cleanup_notes(notes, y, sr, part):
    notes = list(notes)
    if not notes:
        return []
    tuning = _tuning(y, sr)
    gram = np.abs(
        librosa.cqt(
            np.asarray(y, dtype=np.float32),
            sr=sr,
            hop_length=HOP,
            fmin=librosa.midi_to_hz(FMIN_MIDI),
            n_bins=N_BINS,
            bins_per_octave=12,
            tuning=tuning,
        )
    )
    attacks = _attacks(y, sr, HOP)
    spec = PARTS[part]
    low = int(round(librosa.hz_to_midi(spec["fmin"])))
    high = int(round(librosa.hz_to_midi(spec["fmax"])))
    kept = []
    curves = []
    for note in notes:
        start = max(0, int(round(note.time * sr / HOP)))
        stop = min(gram.shape[1], max(start + 1, int(round(min(note.end, note.time + 0.35) * sr / HOP))))
        vec = np.mean(gram[:, start:stop], axis=1)
        if float(np.max(vec)) < 1e-8:
            continue
        pitch = align_pitch(note.pitch, vec)
        if part == "bass":
            pitch = lower_octave(pitch, vec, lowest=low)
        pitch = min(high, max(low, int(pitch)))
        if not _supported(pitch, vec, FMIN_MIDI):
            continue
        kept.append(Note(note.time, note.end, pitch, note.velocity, note.part))
        curves.append(vec)
    cleaned = suppress_harmonics(kept, curves)
    return _merge_overlaps([_snap_start(note, attacks) for note in cleaned])


def transcribe_mono(y, sr, part, fmin, fmax, tuning=0.0):
    f0, voiced, prob = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, hop_length=HOP)
    midi = np.full(len(f0), -1, dtype=int)
    good = voiced & np.isfinite(f0) & (prob >= 0.45)
    midi[good] = np.round(librosa.hz_to_midi(f0[good]) - tuning).astype(int)
    for index in range(1, len(midi) - 1):
        if midi[index - 1] >= 0 and midi[index + 1] >= 0 and abs(int(midi[index]) - int(midi[index - 1])) == 12 and abs(int(midi[index + 1]) - int(midi[index - 1])) <= 1:
            midi[index] = midi[index - 1]
    attack_frames = {int(librosa.time_to_frames(time, sr=sr, hop_length=HOP)) for time in _attacks(y, sr, HOP)}
    min_frames = max(2, int(0.08 * sr / HOP))
    notes = []
    index = 0
    while index < len(midi):
        if midi[index] < 0:
            index += 1
            continue
        pitches = [int(midi[index])]
        stop = index + 1
        while stop < len(midi):
            if stop in attack_frames and stop - index >= min_frames:
                break
            if midi[stop] < 0:
                look = stop
                while look < len(midi) and midi[look] < 0 and look - stop < 3:
                    look += 1
                if look < len(midi) and midi[look] >= 0 and look not in attack_frames and abs(int(midi[look]) - int(np.median(pitches))) <= 1:
                    stop = look
                    continue
                break
            if abs(int(midi[stop]) - int(np.median(pitches))) > 1:
                break
            pitches.append(int(midi[stop]))
            stop += 1
        start = float(librosa.frames_to_time(index, sr=sr, hop_length=HOP))
        end = float(librosa.frames_to_time(stop, sr=sr, hop_length=HOP))
        if end - start >= 0.08:
            segment = y[int(start * sr) : max(int(start * sr) + 1, int(end * sr))]
            rms = float(np.sqrt(np.mean(segment * segment))) if segment.size else 0.0
            notes.append(Note(round(start, 3), round(end, 3), int(np.round(np.median(pitches))), _velocity(min(rms / 0.2, 1.0)), part))
        index = stop
    return notes


def transcribe_poly(path, part, fmin, fmax):
    spec = PARTS[part]
    try:
        logging.disable(logging.WARNING)
        from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
        from basic_pitch.inference import predict
    except ImportError as exc:
        raise SystemExit("缺少 basic-pitch。请执行: pip install basic-pitch==0.4.0 --no-deps") from exc
    finally:
        logging.disable(logging.NOTSET)
    model = build_icassp_2022_model_path(FilenameSuffix.onnx)
    with contextlib.redirect_stdout(io.StringIO()):
        _, _, events = predict(
            Path(path),
            model_or_model_path=model,
            onset_threshold=spec["onset"],
            frame_threshold=spec["frame"],
            minimum_note_length=spec["min_ms"],
            minimum_frequency=fmin,
            maximum_frequency=fmax,
        )
    notes = []
    for start, end, pitch, amplitude, _ in events:
        if float(end) - float(start) < 0.05:
            continue
        notes.append(Note(round(float(start), 3), round(float(end), 3), int(pitch), _velocity(float(amplitude)), part))
    return notes


def transcribe_part(path, part, sr=22050):
    spec = PARTS[part]
    y, file_sr = librosa.load(path, sr=sr, mono=True)
    if y.size < file_sr // 10 or float(np.max(np.abs(y))) < 1e-4:
        return []
    if spec["poly"]:
        notes = transcribe_poly(path, part, spec["fmin"], spec["fmax"])
    else:
        notes = transcribe_mono(y, file_sr, part, spec["fmin"], spec["fmax"], _tuning(y, file_sr))
    return cleanup_notes(notes, y, file_sr, part)
