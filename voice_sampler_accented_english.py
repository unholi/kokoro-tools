from kokoro import KPipeline
import soundfile as sf
import numpy as np
import os
import re

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

VOICES_FILE = "voices.txt"
OUTPUT_DIR  = "./voice_samples_accented_english"
SPEED       = 1.0
LANG_CODE   = "a"   # American English pipeline for all voices → accented English

# Body of sample text appended after the personalised intro
SAMPLE_BODY = (
    " This short passage is written for audiobook narration testing."
    " I can be used to narrate your audiobook project with my unique accent."
    " The goal is simply to hear how the voice handles steady pacing and clear sentence structure."
    " As you listen, notice how each line flows naturally from one thought to the next."
)

# Nationality label from lang char (first letter of prefix)
NATIONALITY_MAP = {
    "a": "American",   "b": "British",    "e": "Spanish",
    "f": "French",     "h": "Hindi",      "i": "Italian",
    "j": "Japanese",   "p": "Portuguese", "z": "Mandarin",
}

# Skip custom blended voices — they already have their own personalised sample
SKIP_IDS = {"cm_voice", "cf_voice"}

# ─────────────────────────────────────────────


def voice_intro(voice_id: str) -> str:
    """Build personalised intro from voice ID.
    af_bella  → 'Hello, my name is Bella. I am an American female voice.'
    bm_george → 'Hello, my name is George. I am a British male voice.'
    """
    prefix      = voice_id[:2]
    lang_char   = prefix[0]
    gender_char = prefix[1]
    name        = voice_id[3:].replace("_", " ").title() if len(voice_id) > 3 else voice_id
    nationality = NATIONALITY_MAP.get(lang_char, "unknown")
    gender      = "male" if gender_char == "m" else "female"
    article     = "an" if nationality[0].lower() in "aeiou" else "a"
    return f"Hello, my name is {name}. I am {article} {nationality} {gender} voice."


def build_sample_text(voice_id: str) -> str:
    return voice_intro(voice_id) + SAMPLE_BODY


def parse_voices(filepath: str) -> list[str]:
    voices = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.lower().startswith("upload"):
                continue
            name = os.path.basename(line)
            name = re.sub(r"\.pt$", "", name, flags=re.IGNORECASE)
            if re.match(r"^[a-z]{2}_[a-z0-9_]+$", name):
                voices.append(name)
    return voices


def try_generate(pipeline: KPipeline, voice_id: str, out_path: str) -> bool:
    text   = build_sample_text(voice_id)
    chunks = []
    try:
        for _, _, audio in pipeline(text, voice=voice_id, speed=SPEED, split_pattern=r'\n+'):
            chunks.append(audio)
        if chunks:
            sf.write(out_path, np.concatenate(chunks), 24000)
            return True
        return False
    except Exception as e:
        print(f"    ✗ error: {e}")
        return False


def main(voices_file=VOICES_FILE, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    voices = [v for v in parse_voices(voices_file) if v not in SKIP_IDS]
    if not voices:
        print(f"No voices found in {voices_file}. Exiting.")
        return

    print(f"Kokoro Accented English Voice Sampler")
    print(f"  All voices read English — foreign voices produce accented English")
    print(f"  Voices file : {voices_file}")
    print(f"  Output dir  : {os.path.abspath(output_dir)}")
    print(f"  Voices found: {len(voices)}\n")

    pipeline = KPipeline(lang_code=LANG_CODE)

    success_count = fail_count = 0
    skipped = []

    for i, voice_id in enumerate(voices, start=1):
        out_path = os.path.join(output_dir, f"{voice_id}.wav")
        if os.path.exists(out_path):
            print(f"  [skip] {voice_id}.wav  (already exists)")
            success_count += 1
            continue

        print(f"  [{i}/{len(voices)}] {voice_id}  → {voice_id}.wav")
        if try_generate(pipeline, voice_id, out_path):
            print(f"    ✓ saved  ({sf.info(out_path).duration:.1f}s)")
            success_count += 1
        else:
            print(f"    ✗ failed — voice may not be cached locally yet")
            skipped.append(voice_id)
            fail_count += 1

    print("\n" + "═" * 50)
    print(f"Done.  ✓ {success_count} succeeded   ✗ {fail_count} failed")
    if skipped:
        print(f"\nFailed voices (not cached locally):")
        for v in skipped:
            print(f"  {v}")
    print(f"\nSamples saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    import sys
    vf  = sys.argv[1] if len(sys.argv) > 1 else VOICES_FILE
    out = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
    main(voices_file=vf, output_dir=out)
