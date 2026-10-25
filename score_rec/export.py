import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mido

from score_rec.spec import DRUM_ORDER, DRUMS, PARTS, note_name

BPM = 120
TPB = 480
GRID = 60
MEASURE = 4 * TPB
STEPS = (1920, 960, 480, 240, 120, 60)
TYPES = {1920: "whole", 960: "half", 480: "quarter", 240: "eighth", 120: "16th", 60: "32nd"}
PITCH_STEPS = "C C# D D# E F F# G G# A A# B".split()
DRUM_HEAD = {
    36: ("F", 4, False),
    38: ("C", 5, False),
    47: ("A", 4, False),
    42: ("G", 5, True),
    46: ("G", 5, True),
    51: ("D", 5, True),
    49: ("A", 5, True),
}


def _ticks(seconds):
    return int(round(seconds * BPM * TPB / 60))


def _snap(tick):
    return int(round(tick / GRID) * GRID)


def _spell(duration):
    pieces = []
    left = duration
    for size in STEPS:
        while left >= size:
            pieces.append(size)
            left -= size
    return pieces


def _child(parent, tag, text=None, **attrib):
    element = ET.SubElement(parent, tag, {key: str(value) for key, value in attrib.items()})
    if text is not None:
        element.text = str(text)
    return element


def _quantize(events):
    chosen = {}
    for note_id, (start, end, pitch, velocity) in enumerate(events):
        start = max(0, _snap(start))
        end = max(start + GRID, _snap(end))
        key = (pitch, start)
        current = chosen.get(key)
        if current is None or end > current[1]:
            chosen[key] = (start, end, pitch, velocity, note_id)
    return list(chosen.values())


def _pitched_events(notes):
    return [(_ticks(note.time), _ticks(note.end), note.pitch, note.velocity) for note in notes]


def _drum_events(hits):
    events = []
    for hit in hits:
        spec = DRUMS[hit.drum]
        events.append((_ticks(hit.time), _ticks(hit.time + spec["dur"]), spec["midi"], hit.velocity))
    return events


def _slices(events):
    quantized = _quantize(events)
    end = max((item[1] for item in quantized), default=0)
    total = max(MEASURE, ((end + MEASURE - 1) // MEASURE) * MEASURE)
    points = {0, total}
    for start, stop, _, _, _ in quantized:
        points.add(min(start, total))
        points.add(min(stop, total))
    for tick in range(0, total + 1, MEASURE):
        points.add(tick)
    points = sorted(point for point in points if 0 <= point <= total)
    slices = []
    for left, right in zip(points, points[1:]):
        if right <= left:
            continue
        sounding = {}
        for start, stop, pitch, _, note_id in quantized:
            if start <= left < stop and pitch not in sounding:
                sounding[pitch] = note_id
        cursor = left
        for size in _spell(right - left):
            slices.append((cursor, size, dict(sounding)))
            cursor += size
    return slices, total // MEASURE


def _append_note(parent, pitch, duration, staff, voice, percussion, tie_stop, tie_start, chord):
    note = _child(parent, "note")
    if chord:
        _child(note, "chord")
    if pitch is None:
        _child(note, "rest")
    elif percussion:
        step, octave, _ = DRUM_HEAD.get(pitch, ("C", 5, False))
        unpitched = _child(note, "unpitched")
        _child(unpitched, "display-step", step)
        _child(unpitched, "display-octave", octave)
    else:
        spelled = PITCH_STEPS[pitch % 12]
        sounded = _child(note, "pitch")
        _child(sounded, "step", spelled[0])
        if len(spelled) > 1:
            _child(sounded, "alter", 1)
        _child(sounded, "octave", pitch // 12 - 1)
    _child(note, "duration", duration)
    if tie_stop:
        _child(note, "tie", type="stop")
    if tie_start:
        _child(note, "tie", type="start")
    if percussion and pitch is not None:
        _child(note, "instrument", id=f"drum-{pitch}")
    _child(note, "voice", voice)
    _child(note, "type", TYPES[duration])
    if not percussion and pitch is not None and len(PITCH_STEPS[pitch % 12]) > 1 and not tie_stop:
        _child(note, "accidental", "sharp")
    if percussion and pitch is not None and DRUM_HEAD.get(pitch, ("C", 5, False))[2]:
        _child(note, "notehead", "x")
    if staff:
        _child(note, "staff", staff)
    if tie_stop or tie_start:
        notations = _child(note, "notations")
        if tie_stop:
            _child(notations, "tied", type="stop")
        if tie_start:
            _child(notations, "tied", type="start")


def _groups(sounding, grand):
    if not grand:
        return [(None, 1, sounding)]
    return [
        (1, 1, {pitch: vel for pitch, vel in sounding.items() if pitch >= 60}),
        (2, 2, {pitch: vel for pitch, vel in sounding.items() if pitch < 60}),
    ]


def _emit(parent, slices, index, grand, percussion):
    start, duration, sounding = slices[index]
    previous = slices[index - 1] if index else None
    nxt = slices[index + 1] if index + 1 < len(slices) else None
    for staff, voice, pitches in _groups(sounding, grand):
        ordered = sorted(pitches)
        if not ordered:
            _append_note(parent, None, duration, staff, voice, False, False, False, False)
            continue
        for chord, pitch in enumerate(ordered):
            note_id = sounding[pitch]
            prev_ok = previous is not None and previous[0] + previous[1] == start and previous[2].get(pitch) == note_id
            next_ok = nxt is not None and start + duration == nxt[0] and nxt[2].get(pitch) == note_id
            _append_note(parent, pitch, duration, staff, voice, percussion, prev_ok, next_ok, chord > 0)


def _attributes(parent, clef):
    attributes = _child(parent, "attributes")
    _child(attributes, "divisions", TPB)
    if clef == "grand":
        _child(attributes, "staves", 2)
    key = _child(attributes, "key")
    _child(key, "fifths", 0)
    time = _child(attributes, "time")
    _child(time, "beats", 4)
    _child(time, "beat-type", 4)
    if clef == "grand":
        for number, sign, line in ((1, "G", 2), (2, "F", 4)):
            clef_el = _child(attributes, "clef", number=number)
            _child(clef_el, "sign", sign)
            _child(clef_el, "line", line)
    elif clef == "percussion":
        _child(_child(attributes, "clef"), "sign", "percussion")
    else:
        clef_el = _child(attributes, "clef")
        _child(clef_el, "sign", "F" if clef == "bass" else "G")
        _child(clef_el, "line", 4 if clef == "bass" else 2)


def _part_xml(root, part_id, events, clef, percussion, tempo):
    part = _child(root, "part", id=part_id)
    slices, measures = _slices(events)
    for number in range(measures):
        measure = _child(part, "measure", number=number + 1)
        if number == 0:
            _attributes(measure, clef)
            if tempo:
                direction = _child(measure, "direction", placement="above")
                direction_type = _child(direction, "direction-type")
                metronome = _child(direction_type, "metronome")
                _child(metronome, "beat-unit", "quarter")
                _child(metronome, "per-minute", BPM)
                _child(direction, "sound", tempo=BPM)
        chosen = [item for item in slices if item[0] // MEASURE == number]
        if not chosen:
            _emit(measure, [(number * MEASURE, MEASURE, {})], 0, clef == "grand", percussion)
            continue
        offset = slices.index(chosen[0])
        for index in range(len(chosen)):
            _emit(measure, slices, offset + index, clef == "grand", percussion)


def write_musicxml(notes, hits, path, title="score"):
    root = ET.Element("score-partwise", version="3.1")
    work = _child(root, "work")
    _child(work, "work-title", Path(title).name)
    part_list = _child(root, "part-list")
    blocks = []
    for part, spec in PARTS.items():
        own = [note for note in notes if note.part == part]
        if own:
            blocks.append((part, spec["name"], _pitched_events(own), spec["clef"], False, spec["program"] + 1, spec["channel"] + 1))
    if hits:
        blocks.append(("drums", "鼓", _drum_events(hits), "percussion", True, 1, 10))
    if not blocks:
        blocks.append(("empty", "空", [], "treble", False, 1, 1))
    for index, (key, name, events, clef, percussion, program, channel) in enumerate(blocks, start=1):
        score_part = _child(part_list, "score-part", id=f"P{index}")
        _child(score_part, "part-name", name)
        if percussion:
            drum_names = {spec["midi"]: spec["name"] for spec in DRUMS.values()}
            for midi, drum in drum_names.items():
                instrument = _child(score_part, "score-instrument", id=f"drum-{midi}")
                _child(instrument, "instrument-name", drum)
                midi_instrument = _child(score_part, "midi-instrument", id=f"drum-{midi}")
                _child(midi_instrument, "midi-channel", 10)
                _child(midi_instrument, "midi-program", 1)
                _child(midi_instrument, "midi-unpitched", midi)
        else:
            instrument = _child(score_part, "score-instrument", id=f"P{index}-I1")
            _child(instrument, "instrument-name", name)
            midi_instrument = _child(score_part, "midi-instrument", id=f"P{index}-I1")
            _child(midi_instrument, "midi-channel", channel)
            _child(midi_instrument, "midi-program", program)
        _part_xml(root, f"P{index}", events, clef, percussion, index == 1)
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    path = Path(path)
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def _append_events(track, channel, events):
    messages = []
    for start, end, note, velocity in events:
        end = max(start + 1, end)
        messages.append((start, 1, note, velocity))
        messages.append((end, 0, note, 0))
    messages.sort()
    last = 0
    for tick, on, note, velocity in messages:
        kind = "note_on" if on else "note_off"
        track.append(mido.Message(kind, channel=channel, note=note, velocity=velocity, time=tick - last))
        last = tick


def _track(mid, name, channel, program, events):
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name=name, time=0))
    if program is not None:
        track.append(mido.Message("program_change", channel=channel, program=program, time=0))
    _append_events(track, channel, events)
    return track


def write_part_midi(notes, path, part):
    spec = PARTS[part]
    mid = mido.MidiFile(ticks_per_beat=TPB)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(BPM), time=0))
    track.append(mido.MetaMessage("track_name", name=part, time=0))
    track.append(mido.Message("program_change", channel=spec["channel"], program=spec["program"], time=0))
    _append_events(track, spec["channel"], _pitched_events(notes))
    path = Path(path)
    mid.save(path)
    return path


def write_score_midi(notes, hits, path):
    mid = mido.MidiFile(ticks_per_beat=TPB)
    tempo = mido.MidiTrack()
    mid.tracks.append(tempo)
    tempo.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(BPM), time=0))
    for part, spec in PARTS.items():
        own = [note for note in notes if note.part == part]
        if own:
            _track(mid, part, spec["channel"], spec["program"], _pitched_events(own))
    if hits:
        _track(mid, "drums", 9, None, _drum_events(hits))
    path = Path(path)
    mid.save(path)
    return path


def _pad(text, width):
    used = sum(2 if ord(ch) > 127 else 1 for ch in text)
    return text + " " * max(0, width - used)


def format_timeline(hits, notes=None):
    counts = {}
    for hit in hits:
        counts[hit.drum] = counts.get(hit.drum, 0) + 1
    head = "  ".join(f"{DRUMS[drum]['name']} {counts.get(drum, 0)}" for drum in DRUM_ORDER)
    lines = [head, ""]
    for hit in hits:
        lines.append(f"{hit.time:8.3f}  {_pad(DRUMS[hit.drum]['name'], 8)}{hit.velocity:3d}")
    notes = list(notes or [])
    if notes:
        part_counts = {part: 0 for part in PARTS}
        for note in notes:
            part_counts[note.part] = part_counts.get(note.part, 0) + 1
        lines.extend(["", "  ".join(f"{PARTS[part]['name']} {part_counts[part]}" for part in PARTS), ""])
        for note in sorted(notes, key=lambda item: (item.time, item.part, item.pitch)):
            lines.append(
                f"{note.time:8.3f}  {_pad(PARTS[note.part]['name'], 8)}{note_name(note.pitch):<4}{note.velocity:3d}  {note.end:8.3f}"
            )
    return "\n".join(lines) + "\n"


def write_timeline(hits, source, out_dir, notes=None):
    out_dir = Path(out_dir)
    text_path = out_dir / "timeline.txt"
    json_path = out_dir / "timeline.json"
    text_path.write_text(format_timeline(hits, notes), encoding="utf-8")
    payload = {
        "source": str(source),
        "bpm": BPM,
        "time_signature": "4/4",
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
        "notes": [
            {
                "time": note.time,
                "end": note.end,
                "part": note.part,
                "name": PARTS[note.part]["name"],
                "pitch": note.pitch,
                "note": note_name(note.pitch),
                "velocity": note.velocity,
            }
            for note in notes or []
        ],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return text_path, json_path


def write_drum_midi(hits, path, bpm=BPM):
    mid = mido.MidiFile(ticks_per_beat=TPB)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    scale = bpm * TPB / 60
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

