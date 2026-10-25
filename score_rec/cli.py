import argparse
import sys
from pathlib import Path

from score_rec.pipeline import transcribe


def build_parser():
    parser = argparse.ArgumentParser(description="从混音中分轨，识别鼓和有音高乐器，导出 MIDI 与乐谱")
    parser.add_argument("audio", type=Path, help="音乐文件，例如 mp3、wav、flac")
    parser.add_argument("-o", "--output", type=Path, help="输出目录，默认 score_rec_out/<文件名>")
    parser.add_argument("--no-stems", action="store_true", help="不保留分离出的音频")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    audio = args.audio.expanduser().resolve()
    out_dir = args.output or Path("score_rec_out") / audio.stem
    try:
        transcribe(audio, out_dir, keep_stems=not args.no_stems)
    except BrokenPipeError:
        sys.exit(0)


if __name__ == "__main__":
    main()
