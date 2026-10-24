import tempfile
import unittest
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from score_rec.drums import detect_onsets, find_stem, is_open_hat, transcribe_stems, write_midi, write_timeline


def tone(sr, decay, tau, at=0.0, length=0.6):
    y = np.zeros(int(length * sr), dtype=np.float32)
    m = int(decay * sr)
    t = np.arange(m, dtype=np.float32) / sr
    i = int(at * sr)
    y[i : i + m] = np.sin(2 * np.pi * 1800 * t) * np.exp(-t / tau)
    y *= 0.8 / max(float(np.max(np.abs(y))), 1e-8)
    return y


class DrumTests(unittest.TestCase):
    def test_find_stem(self):
        paths = ["song_(Drums)_htdemucs_ft.wav", "kick.wav"]
        self.assertEqual(find_stem(paths, ["drums"]).name, "song_(Drums)_htdemucs_ft.wav")
        self.assertEqual(find_stem(paths, ["kick"]).name, "kick.wav")
        self.assertIsNone(find_stem(paths, ["ride"]))

    def test_onsets_and_hats(self):
        sr = 22050
        y = np.zeros(int(3 * sr), dtype=np.float32)
        rng = np.random.default_rng(1)
        for t in (0.40, 1.10, 2.00):
            i = int(t * sr)
            m = int(0.03 * sr)
            env = np.exp(-np.linspace(0, 7, m)).astype(np.float32)
            y[i : i + m] += rng.standard_normal(m).astype(np.float32) * env
        y *= 0.8 / np.max(np.abs(y))
        found = [t for t, _ in detect_onsets(y, sr, 0.05)]
        self.assertEqual(len(found), 3)
        for expected, actual in zip((0.40, 1.10, 2.00), found):
            self.assertLess(abs(actual - expected), 0.05)

        self.assertFalse(is_open_hat(tone(sr, 0.03, 0.008), sr, 0.0, None))
        self.assertTrue(is_open_hat(tone(sr, 0.40, 0.20), sr, 0.0, None))

    def test_timeline_and_midi(self):
        sr = 22050
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            kick = root / "kick.wav"
            hh = root / "hh.wav"
            sf.write(kick, tone(sr, 0.04, 0.01, at=0.40, length=1.2), sr)
            sf.write(hh, tone(sr, 0.40, 0.20, at=0.40, length=1.2), sr)
            hits = transcribe_stems({"kick": kick, "hh": hh}, sr=sr)
            self.assertTrue(any(hit.drum == "kick" for hit in hits))
            self.assertTrue(any(hit.drum == "hh_open" for hit in hits))
            text_path, _ = write_timeline(hits, "song.wav", root)
            self.assertIn("底鼓", text_path.read_text(encoding="utf-8"))
            midi_path = write_midi(hits, root / "drums.mid")
            notes = []
            tick = 0
            for msg in mido.MidiFile(midi_path).tracks[0]:
                tick += msg.time
                if msg.type == "note_on" and msg.velocity and msg.channel == 9:
                    notes.append(msg.note)
            self.assertIn(36, notes)
            self.assertIn(46, notes)


if __name__ == "__main__":
    unittest.main()
