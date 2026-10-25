from dataclasses import dataclass

import librosa

FT_MODEL = "htdemucs_ft.yaml"
SIX_MODEL = "htdemucs_6s.yaml"
KIT_MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
FT_STEMS = ("vocals", "drums", "bass")
SIX_STEMS = ("guitar", "piano", "other")
DEMUCS = {"segment_size": "Default", "shifts": 2, "overlap": 0.25, "segments_enabled": True}
KIT_PARAMS = {
    "segment_size": 256,
    "override_model_segment_size": False,
    "batch_size": 1,
    "overlap": 2,
    "pitch_shift": 0,
    "process_all_stems": True,
}
KIT = {
    "kick": ("kick",),
    "snare": ("snare",),
    "toms": ("toms", "tom"),
    "hh": ("hh", "hihat", "hi-hat"),
    "ride": ("ride",),
    "crash": ("crash",),
}

DRUMS = {
    "kick": {"name": "底鼓", "midi": 36, "gap": 0.06, "dur": 0.10},
    "snare": {"name": "军鼓", "midi": 38, "gap": 0.05, "dur": 0.10},
    "hh_closed": {"name": "闭镲", "midi": 42, "gap": 0.04, "dur": 0.06},
    "hh_open": {"name": "开镲", "midi": 46, "gap": 0.04, "dur": 0.28},
    "toms": {"name": "通鼓", "midi": 47, "gap": 0.05, "dur": 0.16},
    "ride": {"name": "叮叮镲", "midi": 51, "gap": 0.05, "dur": 0.18},
    "crash": {"name": "碎音镲", "midi": 49, "gap": 0.12, "dur": 0.45},
}
DRUM_ORDER = ("kick", "snare", "hh_closed", "hh_open", "toms", "ride", "crash")
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _hz(note):
    return float(librosa.note_to_hz(note))


def _part(name, low, high, poly, program, channel, clef, onset=0.5, frame=0.3, min_ms=80):
    return {
        "name": name,
        "fmin": _hz(low),
        "fmax": _hz(high),
        "poly": poly,
        "program": program,
        "channel": channel,
        "clef": clef,
        "onset": onset,
        "frame": frame,
        "min_ms": min_ms,
    }


PARTS = {
    "vocals": _part("人声", "C3", "C6", False, 53, 0, "treble"),
    "guitar": _part("吉他", "E2", "E6", True, 25, 1, "treble", onset=0.5, frame=0.34, min_ms=100),
    "piano": _part("钢琴", "A0", "C8", True, 0, 2, "grand", onset=0.52, frame=0.36, min_ms=110),
    "bass": _part("贝斯", "E1", "C4", False, 32, 3, "bass"),
}


@dataclass(frozen=True)
class Hit:
    time: float
    drum: str
    velocity: int


@dataclass(frozen=True)
class Note:
    time: float
    end: float
    pitch: int
    velocity: int
    part: str


def note_name(pitch):
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"
