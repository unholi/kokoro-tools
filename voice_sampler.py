from kokoro import KPipeline
import soundfile as sf
import numpy as np
import os
import re

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

VOICES_FILE = "voices.txt"
OUTPUT_DIR  = "./voice_samples"
SPEED       = 1.0

# Body of sample text appended after the personalised intro (English voices)
SAMPLE_BODY = (
    " This short passage is written for audiobook narration testing."
    " The goal is simply to hear how the voice handles steady pacing and clear sentence structure."
    " As you listen, notice how each line flows naturally from one thought to the next."
)

# ─────────────────────────────────────────────
# Voice prefix → Kokoro lang_code mapping
LANG_MAP = {
    "af": "a",  "am": "a",
    "bf": "b",  "bm": "b",
    "ef": "e",  "em": "e",
    "ff": "f",
    "hf": "h",  "hm": "h",
    "if": "i",  "im": "i",
    "jf": "j",  "jm": "j",
    "pf": "p",  "pm": "p",
    "zf": "z",  "zm": "z",
}

# Nationality label from lang char
NATIONALITY_MAP = {
    "a": "American",   "b": "British",    "e": "Spanish",
    "f": "French",     "h": "Hindi",      "i": "Italian",
    "j": "Japanese",   "p": "Portuguese", "z": "Mandarin",
}

# Native-language body text for non-English voices
LANG_BODY_SAMPLES = {
    "j": "こんにちは！これは私の声のサンプルです。声のペースと明瞭さにご注目ください。",
    "z": "您好。这是我的声音样本。请注意声音的节奏和清晰度。",
    "f": "Bonjour ! Voici un exemple de ma voix. Remarquez le rythme et la clarté de l'élocution.",
    "e": "¡Hola! Esta es una muestra de mi voz. Observe el ritmo y la claridad de la narración.",
    "p": "Olá! Este é um exemplo da minha voz. Observe o ritmo e a clareza da narração.",
    "i": "Ciao! Questo è un campione della mia voce. Si prega di notare il ritmo e la chiarezza.",
    "h": "नमस्ते! यह मेरी आवाज़ का एक नमूना है। कृपया गति और स्पष्टता पर ध्यान दें।",
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


def get_lang_code(voice_id: str) -> str:
    return LANG_MAP.get(voice_id[:2], "a")


def build_sample_text(voice_id: str, lang_code: str) -> str:
    intro = voice_intro(voice_id)
    body  = LANG_BODY_SAMPLES.get(lang_code, SAMPLE_BODY)
    return intro + body


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


def try_generate(pipeline: KPipeline, voice_id: str, lang_code: str, out_path: str) -> bool:
    text   = build_sample_text(voice_id, lang_code)
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

    print(f"Kokoro Voice Sampler (native language)")
    print(f"  Voices file : {voices_file}")
    print(f"  Output dir  : {os.path.abspath(output_dir)}")
    print(f"  Voices found: {len(voices)}\n")

    by_lang: dict[str, list[str]] = {}
    for v in voices:
        by_lang.setdefault(get_lang_code(v), []).append(v)

    success_count = fail_count = 0
    skipped = []

    for lang_code, lang_voices in sorted(by_lang.items()):
        print(f"── lang_code='{lang_code}'  ({len(lang_voices)} voice(s)) ──")
        try:
            pipeline = KPipeline(lang_code=lang_code)
        except Exception as e:
            print(f"  Could not load pipeline for lang_code='{lang_code}': {e}")
            skipped.extend(lang_voices)
            fail_count += len(lang_voices)
            continue

        for voice_id in lang_voices:
            out_path = os.path.join(output_dir, f"{voice_id}.wav")
            if os.path.exists(out_path):
                print(f"  [skip] {voice_id}.wav  (already exists)")
                success_count += 1
                continue

            print(f"  [{success_count + fail_count + 1}/{len(voices)}] {voice_id}  → {voice_id}.wav")
            if try_generate(pipeline, voice_id, lang_code, out_path):
                print(f"    ✓ saved  ({sf.info(out_path).duration:.1f}s)")
                success_count += 1
            else:
                print(f"    ✗ failed — voice may not be cached locally yet")
                skipped.append(voice_id)
                fail_count += 1
        print()

    print("═" * 50)
    print(f"Done.  ✓ {success_count} succeeded   ✗ {fail_count} failed")
    if skipped:
        print(f"\nFailed voices:")
        for v in skipped:
            print(f"  {v}")
    print(f"\nSamples saved to: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    import sys
    vf  = sys.argv[1] if len(sys.argv) > 1 else VOICES_FILE
    out = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
    main(voices_file=vf, output_dir=out)
