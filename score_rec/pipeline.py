import os
import shutil
from pathlib import Path

from score_rec.drums import transcribe_stems
from score_rec.export import write_drum_midi, write_musicxml, write_part_midi, write_score_midi, write_timeline
from score_rec.pitch import transcribe_part
from score_rec.separate import ensure_ffmpeg, split_kit, split_mix, workspace_dir
from score_rec.spec import PARTS


def transcribe(audio, out_dir, keep_stems=True):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    audio = Path(audio).expanduser().resolve()
    if not audio.is_file():
        raise SystemExit(f"找不到音频文件: {audio}")
    ensure_ffmpeg()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("未找到 ffmpeg。")

    out_dir = Path(out_dir).resolve()
    stem_dir = out_dir / "stems"
    model_dir = workspace_dir() / "models"
    stem_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out_dir}")

    stems = split_mix(audio, stem_dir, model_dir)

    print("3/4 拆分底鼓、军鼓、踩镲、通鼓、叮叮镲、碎音镲")
    kit = split_kit(stems["drums"], stem_dir, model_dir)

    print("4/4 识别音符并校对音高")
    hits = transcribe_stems(kit)
    notes = []
    for name in PARTS:
        notes.extend(transcribe_part(stems[name], name))

    print("导出时间轴、MIDI 和乐谱")
    text_path, json_path = write_timeline(hits, audio, out_dir, notes)
    drum_midi = write_drum_midi(hits, out_dir / "drums.mid")
    score_midi = write_score_midi(notes, hits, out_dir / "score.mid")
    musicxml = write_musicxml(notes, hits, out_dir / "score.musicxml", audio.name)
    part_midis = []
    for name in PARTS:
        own = [note for note in notes if note.part == name]
        if own:
            part_midis.append(write_part_midi(own, out_dir / f"{name}.mid", name))
    if not keep_stems:
        shutil.rmtree(stem_dir, ignore_errors=True)

    print(text_path.read_text(encoding="utf-8").splitlines()[0])
    print(f"时间轴: {text_path}")
    print(f"JSON: {json_path}")
    print(f"鼓 MIDI: {drum_midi}")
    print(f"总谱 MIDI: {score_midi}")
    print(f"乐谱: {musicxml}")
    for path in part_midis:
        print(f"分轨 MIDI: {path}")
