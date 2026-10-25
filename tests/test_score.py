import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from score_rec.export import write_musicxml, write_part_midi, write_score_midi, write_timeline
from score_rec.pitch import cleanup_notes, transcribe_part
from score_rec.spec import Hit, Note


def _sine(sr, freq, seconds=1.0):
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    y = 0.4 * np.sin(2 * np.pi * freq * t)
    edge = int(0.02 * sr)
    y[:edge] *= np.linspace(0, 1, edge, dtype=np.float32)
    y[-edge:] *= np.linspace(1, 0, edge, dtype=np.float32)
    return y


def _measure_duration(measure, staff=None):
    total = 0
    for note in measure.findall("note"):
        if note.find("chord") is not None:
            continue
        if staff is not None:
            found = note.find("staff")
            if found is None or found.text != str(staff):
                continue
        total += int(note.findtext("duration"))
    return total


class ScoreTests(unittest.TestCase):
    def test_mono_and_poly(self):
        sr = 22050
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "a4.wav"
            sf.write(path, _sine(sr, 440), sr)
            mono = transcribe_part(path, "vocals", sr=sr)
            poly = transcribe_part(path, "guitar", sr=sr)
            self.assertTrue(any(note.pitch == 69 and note.end - note.time > 0.5 for note in mono))
            self.assertTrue(any(note.pitch == 69 for note in poly))
            silent = Path(temp) / "silent.wav"
            sf.write(silent, np.zeros(sr, dtype=np.float32), sr)
            self.assertEqual(transcribe_part(silent, "piano", sr=sr), [])

    def test_repeated_notes_and_pitch_check(self):
        sr = 22050
        count = int(1.6 * sr)
        times = np.arange(count) / sr
        level = np.full(count, 0.12, dtype=np.float32)
        level[(times >= 0.15) & (times < 0.65)] = 0.8
        level[(times >= 0.90) & (times < 1.40)] = 0.85
        sung = (level * np.sin(2 * np.pi * 440 * times)).astype(np.float32)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "repeat.wav"
            sf.write(path, sung, sr)
            notes = transcribe_part(path, "vocals", sr=sr)
        pitches = [note.pitch for note in notes]
        self.assertGreaterEqual(pitches.count(69), 2)

        fundamental = (0.4 * np.sin(2 * np.pi * 440 * times[:sr])).astype(np.float32)
        harmonic = fundamental + (0.08 * np.sin(2 * np.pi * 880 * times[:sr])).astype(np.float32)
        paired = [
            Note(0.1, 0.9, 69, 100, "piano"),
            Note(0.1, 0.9, 81, 40, "piano"),
        ]
        cleaned = cleanup_notes(paired, harmonic, sr, "piano")
        self.assertEqual([note.pitch for note in cleaned], [69])

        wrong = cleanup_notes([Note(0.1, 0.9, 70, 90, "piano")], fundamental, sr, "piano")
        self.assertEqual([note.pitch for note in wrong], [69])

        bass_tone = (0.5 * np.sin(2 * np.pi * 55 * times[:sr])).astype(np.float32)
        bass = cleanup_notes([Note(0.1, 0.9, 45, 90, "bass")], bass_tone, sr, "bass")
        self.assertEqual([note.pitch for note in bass], [33])

    def test_timeline_musicxml_and_midi(self):
        notes = [
            Note(0.0, 2.5, 60, 90, "vocals"),
            Note(0.0, 0.5, 64, 70, "vocals"),
            Note(0.0, 1.0, 60, 80, "piano"),
            Note(0.0, 1.0, 48, 80, "piano"),
            Note(0.0, 0.5, 40, 88, "bass"),
        ]
        hits = [Hit(0.0, "kick", 100), Hit(0.0, "hh_closed", 80)]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            text_path, json_path = write_timeline(hits, "song.wav", root, notes)
            text = text_path.read_text(encoding="utf-8")
            self.assertIn("底鼓", text)
            self.assertIn("人声", text)
            self.assertIn("C4", text)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["bpm"], 120)
            self.assertEqual(payload["notes"][0]["pitch"], 60)

            xml_path = write_musicxml(notes, hits, root / "score.musicxml", "song.wav")
            score = ET.parse(xml_path).getroot()
            names = [item.findtext("part-name") for item in score.find("part-list")]
            self.assertEqual(names, ["人声", "钢琴", "贝斯", "鼓"])
            for part in score.findall("part"):
                staves = {note.findtext("staff") for note in part.findall(".//note")}
                if "2" in staves:
                    for measure in part.findall("measure"):
                        self.assertEqual(_measure_duration(measure, "1"), 1920)
                        self.assertEqual(_measure_duration(measure, "2"), 1920)
                else:
                    for measure in part.findall("measure"):
                        self.assertEqual(_measure_duration(measure), 1920)
            vocal = score.find("part")
            pitched = [note.find("pitch") for note in vocal.find("measure").findall("note") if note.find("pitch") is not None]
            steps = [(item.findtext("step"), item.findtext("octave")) for item in pitched]
            self.assertIn(("C", "4"), steps)
            self.assertIn(("E", "4"), steps)
            self.assertIsNotNone(vocal.find(".//chord"))
            self.assertIsNotNone(vocal.find(".//tie"))
            repeated = write_musicxml(
                [Note(0.0, 0.5, 60, 90, "vocals"), Note(0.5, 1.0, 60, 90, "vocals")],
                [],
                root / "repeated.musicxml",
            )
            repeated_score = ET.parse(repeated).getroot()
            self.assertIsNone(repeated_score.find(".//tie"))
            drum = score.findall("part")[-1]
            self.assertIsNotNone(drum.find(".//unpitched"))
            self.assertEqual(drum.find(".//notehead").text, "x")

            score_midi = write_score_midi(notes, hits, root / "score.mid")
            vocal_midi = write_part_midi([note for note in notes if note.part == "vocals"], root / "vocals.mid", "vocals")
            found = set()
            for path in (score_midi, vocal_midi):
                for track in mido.MidiFile(path).tracks:
                    for msg in track:
                        if msg.type == "note_on" and msg.velocity:
                            found.add((msg.channel, msg.note))
            self.assertIn((0, 60), found)
            self.assertIn((0, 64), found)
            self.assertIn((2, 60), found)
            self.assertIn((2, 48), found)
            self.assertIn((3, 40), found)
            self.assertIn((9, 36), found)
            self.assertIn((9, 42), found)


if __name__ == "__main__":
    unittest.main()
