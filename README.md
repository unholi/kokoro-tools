# Kokoro Audiobook Studio

> A local, offline, browser-based tool for converting text files into audiobook-quality narration using the [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) engine.

**No cloud API. No subscription. No data leaving your machine.** After a one-time setup, the studio runs entirely offline.

---

## What's in this package

| File | Required | Purpose |
|------|----------|---------|
| `kokoro-studio` | ✅ Required | Shell launcher — run this to start the app |
| `serve.py` | ✅ Required | Local web server (port 7891) |
| `kokoro_run.py` | ✅ Required | TTS worker — converts text to WAV |
| `kokoro_ui.html` | ✅ Required | Browser interface |
| `voices.txt` | ✅ Required | List of available Kokoro voice files |
| `voice_sampler.py` | Optional | Generates native-language voice previews |
| `voice_sampler_accented_english.py` | Optional | Generates accented-English voice previews |
| `kokoro_audiobook.py` | Optional | Standalone CLI batch converter (no UI needed) |
| `user_guide.html` | Optional | Full in-app user guide (opens from the ? Help button) |
| `README.md` | Optional | This file |

---

## Getting Started

### Step 1 — Install Kokoro and its dependencies

You need Python 3.10+, `espeak-ng`, and a virtual environment with Kokoro installed. The studio expects the venv at `~/Downloads/kokoro/env1/` by default.

```bash
sudo apt install espeak-ng
mkdir -p ~/Downloads/kokoro && cd ~/Downloads/kokoro
python3 -m venv env1
source env1/bin/activate
pip install kokoro soundfile torch --index-url https://download.pytorch.org/whl/cpu
```

> **GPU acceleration (optional):** Replace the last line with:
> ```bash
> pip install kokoro soundfile torch --index-url https://download.pytorch.org/whl/cu121
> ```

---

### Step 2 — Place the kokoro-tools folder

Copy the entire `kokoro-tools/` folder to:

```
~/.local/bin/kokoro-tools/
```

This location is already on `$PATH` for user-installed binaries on Ubuntu 24.04 / Pop!\_OS. If you use a different path, update the `VENV_ACTIVATE` variable inside the `kokoro-studio` script to match your venv location.

---

### Step 3 — Make the launcher executable

```bash
chmod +x ~/.local/bin/kokoro-tools/kokoro-studio
```

You only need to do this once.

---

### Step 4 — Add a shell alias (recommended)

So you can type `kokoro-studio` from anywhere:

```bash
echo "alias kokoro-studio='~/.local/bin/kokoro-tools/kokoro-studio'" >> ~/.bashrc
source ~/.bashrc
```

---

### Step 5 — Run the studio from your project folder

Navigate to the directory containing your `.txt` chapter files, then launch:

```bash
cd ~/my-audiobook-project
kokoro-studio
```

The server starts, your browser opens to **http://localhost:7891**, and the interface is ready. The terminal window that launched the studio must stay open while you work.

---

### Step 6 — Generate voice samples (first time only)

The Voice Audition panel plays pre-generated WAV samples for each voice. Open **Voice Audition** in the UI and click **⚙ Native language** and/or **⚙ Accented English** to generate them. This takes several minutes and only needs to be done once — samples are saved inside `kokoro-tools/` and reused on every future launch.

---

### Step 7 — Prepare your text files

Kokoro reads plain text — no markup, no formatting codes. Punctuation controls pacing:

| Mark | Effect |
|------|--------|
| `,` comma | Brief pause |
| `.` period | Full stop, natural breath |
| `—` em dash | Dramatic mid-sentence break |
| `...` ellipsis | Trailing off, hesitation |

**Spell out numbers and abbreviations:**

```
Use:  "Chapter Three"   not  "Chapter 3"
Use:  "Doctor Voss"     not  "Dr. Voss"
Use:  "twenty-five"     not  "25"
Use:  "and"             not  "&"
```

**Fix curly quotes before converting** (word processors save "smart" quotes that confuse the phonemiser):

```bash
sed -i "s/\u2018/'/g; s/\u2019/'/g; s/\u201C/\"/g; s/\u201D/\"/g" *.txt
```

**Proper nouns and invented names:** Kokoro mispronounces unusual names. Write them phonetically in the source file — e.g. if *Kaelith* should sound like *Kaylith*, use the phonetic spelling. Keep a substitution log so corrections stay consistent across chapters.

**Sentence length:** Sentences over ~40 words can lose natural intonation — break them with a period or em dash. Very short sentences (under 5 words) can sound slightly clipped.

**File encoding:** Save every file as **UTF-8**.

**One chapter per file:** Keep each chapter as a single `.txt` file. Splitting chapters resets Kokoro's prosody (rhythm and intonation) at each boundary, which can make narration sound disjointed at join points.

---

## Switching Project Folders Mid-Session

You do not need to restart the studio to work on a different book or chapter folder. Use the **Folder** bar at the top of the Text Files panel:

```
# Relative to your current folder
../Book2
../Book3/chapters

# Absolute path
/home/ladmin/Documents/AI_Books/Book4
```

Type the path and press **⇄ Switch** or hit Enter. The file table reloads immediately with the new folder's contents. All Settings are preserved. Relative paths resolve against the **current** directory, not the startup directory — so each switch is relative to wherever you currently are.

---

## Platform Notes

> ⚠️ **Ubuntu / Pop!\_OS 24.04:** These instructions are written for Ubuntu 24.04 and Pop!\_OS 24.04. Commands and paths should work as-is on most Debian-based distributions. macOS and Windows users will need to adjust the venv activation path, the `apt` commands, and the terminal emulator detection inside `kokoro-studio`.

> ℹ️ **Fully offline after setup:** Kokoro downloads model weights (~330 MB) and voice files from Hugging Face on first use. After that, no internet connection is required. To lock it offline permanently:
> ```bash
> echo 'export HF_HUB_OFFLINE=1' >> ~/.bashrc
> source ~/.bashrc
> ```

---

## UI Overview

The studio interface is divided into collapsible panels (all accessible from the sticky header):

| Panel | Default | Purpose |
|-------|---------|---------|
| **Voice Audition** | Hidden | Browse and play all 50+ voices; filter by accent and gender; generate sample WAVs |
| **Voice Blender** | Hidden | Mathematically blend two voices into a custom `cf_voice` or `cm_voice` |
| **Settings** | Visible | Speed, language pipeline, and output folder |
| **Sample Audio** | Visible | Generate a spoken test from any text using the active voice |
| **Text Files** | Visible | Select chapters, assign voices, view line/word counts, switch folders, launch conversion |

---

## Key Features

- **50+ voices** across American English, British English, French, Spanish, Hindi, Italian, Japanese, Portuguese, and Mandarin
- **Accented English** — non-English voices speaking English with their natural accent
- **Voice Blender** — create custom blended voices saved as reusable `.pt` tensors
- **Per-file voice assignment** — different voices for different chapters in one launch
- **Sequential batch processing** — multi-voice runs process one voice group at a time, keeping your machine responsive
- **Folder switching** — change source directories mid-session without restarting; supports relative (`../Book2`) and absolute paths
- **Sample Audio panel** — quick spoken test of any text before committing to a full run
- **Session log** — append-only `kokoro_session.log` records every conversion with voice, speed, and duration
- **Fully offline** after initial model download

---

## Resources

| Resource | Link |
|----------|------|
| Kokoro model (Hugging Face) | [huggingface.co/hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) |
| Kokoro Python package (PyPI) | [pypi.org/project/kokoro](https://pypi.org/project/kokoro/) |
| espeak-ng (phonemiser) | [github.com/espeak-ng/espeak-ng](https://github.com/espeak-ng/espeak-ng) |
| PyTorch installation guide | [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) |
| Audacity (free audio editor) | [audacityteam.org](https://www.audacityteam.org) |

---

## Credits

Kokoro Audiobook Studio was designed and built collaboratively between the author and **[Claude](https://claude.ai)** (Anthropic's AI assistant) over an extended development session. The application grew organically from a simple batch conversion script into a full browser-based studio through iterative conversation — each feature, bug fix, and refinement discussed, reasoned through, and implemented turn by turn.

**Built with [Claude](https://claude.ai) by [Anthropic](https://www.anthropic.com)**
Powered by **[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** by hexgrad

### Third-party components

| Component | Author / Maintainer | Licence |
|-----------|---------------------|---------|
| Kokoro-82M model | hexgrad (Hugging Face) | Apache 2.0 |
| kokoro Python package | hexgrad | Apache 2.0 |
| PyTorch | Meta AI / PyTorch Foundation | BSD-style |
| soundfile | Bastian Bechtold | BSD 3-Clause |
| espeak-ng | espeak-ng contributors | GPL v3 |

---

*Ubuntu / Pop!\_OS 24.04 · Fully offline after initial setup*
