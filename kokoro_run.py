#!/usr/bin/env python3
"""
kokoro_run.py  —  Kokoro batch worker
Called by serve.py with a list of .txt files to convert.
Do not run this directly — use the Web UI.
"""

import argparse
import datetime
import os
import sys
from pathlib import Path
import numpy as np


def log_entry(log_path: str, filename: str, voice: str, speed: float, duration_sec: float, status: str):
    """Append a single line to the session log."""
    ts      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    h, rem  = divmod(int(duration_sec), 3600)
    m, s    = divmod(rem, 60)
    dur_str = f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"
    line    = (f"{ts}  {filename:<45}  voice={voice:<20}  "
               f"speed={speed:.2f}x  duration={dur_str}  [{status}]\n")
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files",    nargs="+", required=True)
    parser.add_argument("--voice",    default="af_bella")
    parser.add_argument("--speed",    type=float, default=1.0)
    parser.add_argument("--output",   default="./audiobook_output")
    parser.add_argument("--lang",     default="a")
    parser.add_argument("--work-dir", default=".")
    args = parser.parse_args()

    os.chdir(args.work_dir)
    os.makedirs(args.output, exist_ok=True)

    # Append-only log in the output folder
    log_path = os.path.join(args.output, "kokoro_session.log")

    try:
        from kokoro import KPipeline
        import soundfile as sf
    except ImportError as e:
        print(f"\n[ERROR] Missing dependency: {e}")
        print("Make sure kokoro and soundfile are installed in this Python environment.")
        sys.exit(1)

    print(f"\n{'═'*55}")
    print(f"  Kokoro Audiobook Runner")
    print(f"  Voice    : {args.voice}")
    print(f"  Speed    : {args.speed}x")
    print(f"  Output   : {os.path.abspath(args.output)}")
    print(f"  Files    : {len(args.files)}")
    print(f"{'═'*55}\n")

    # Write session header to log
    ts_start = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n{'─'*90}\n")
        lf.write(f"Session started : {ts_start}\n")
        lf.write(f"Voice           : {args.voice}\n")
        lf.write(f"Speed           : {args.speed}x\n")
        lf.write(f"Output folder   : {os.path.abspath(args.output)}\n")
        lf.write(f"Working dir     : {os.path.abspath(args.work_dir)}\n")
        lf.write(f"Files queued    : {len(args.files)}\n")
        lf.write(f"{'─'*90}\n")

    # Resolve voice — custom blends are stored as tensors, not named voices
    VOICES_PATH = (
        Path.home() / ".cache/huggingface/hub"
        / "models--hexgrad--Kokoro-82M"
        / "snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987/voices"
    )
    CUSTOM_IDS = {"cm_voice", "cf_voice"}
    voice_arg = args.voice
    if voice_arg in CUSTOM_IDS:
        try:
            import torch
            pt_path = VOICES_PATH / f"{voice_arg}.pt"
            voice_arg = torch.load(pt_path, weights_only=True)
            print(f"  Loaded custom tensor: {pt_path}")
        except Exception as e:
            print(f"  [ERROR] Could not load custom voice tensor: {e}")
            sys.exit(1)

    pipeline = KPipeline(lang_code=args.lang)
    silence  = np.zeros(int(24000 * 0.3), dtype=np.float32)

    total_dur = 0.0
    for idx, fname in enumerate(args.files, 1):
        fpath = os.path.join(args.work_dir, fname)
        if not os.path.exists(fpath):
            print(f"[{idx}/{len(args.files)}] SKIP (not found): {fname}")
            log_entry(log_path, fname, args.voice, args.speed, 0, "SKIP - file not found")
            continue

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except UnicodeDecodeError:
            with open(fpath, "r", encoding="latin-1") as f:
                text = f.read().strip()

        if not text:
            print(f"[{idx}/{len(args.files)}] SKIP (empty): {fname}")
            log_entry(log_path, fname, args.voice, args.speed, 0, "SKIP - empty file")
            continue

        base    = os.path.splitext(fname)[0]
        outfile = os.path.join(args.output, base + ".wav")

        print(f"[{idx}/{len(args.files)}] {fname}")
        print(f"  → {outfile}")

        chunks = []
        try:
            gen = pipeline(text, voice=voice_arg, speed=args.speed,
                           split_pattern=r'\n+')
            for i, (gs, ps, audio) in enumerate(gen):
                chunks.append(audio)
                chunks.append(silence)
                preview = gs[:70].strip().replace('\n', ' ')
                print(f"  chunk {i+1}: {preview}{'…' if len(gs)>70 else ''}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            log_entry(log_path, fname, args.voice, args.speed, 0, f"ERROR - {e}")
            continue

        if chunks:
            import soundfile as sf
            combined = np.concatenate(chunks)
            sf.write(outfile, combined, 24000)
            dur = len(combined) / 24000
            total_dur += dur
            m, s = divmod(int(dur), 60)
            log_entry(log_path, fname, args.voice, args.speed, dur, "OK")
            print(f"  ✓ saved ({m}m {s:02d}s)\n")
        else:
            log_entry(log_path, fname, args.voice, args.speed, 0, "WARN - no audio generated")
            print(f"  [WARN] No audio generated\n")

    th, rem = divmod(int(total_dur), 3600)
    tm, ts  = divmod(rem, 60)
    print(f"{'═'*55}")
    print(f"  All done!  Total audio: {th}h {tm:02d}m {ts:02d}s")
    print(f"  Output folder: {os.path.abspath(args.output)}")
    print(f"{'═'*55}")
    # Write session footer
    ts_end = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"Session finished : {ts_end}  |  Total audio: {th}h {tm:02d}m {ts:02d}s\n")

    print(f"  Log file: {log_path}")


if __name__ == "__main__":
    main()
