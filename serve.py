#!/usr/bin/env python3
"""
serve.py  —  Kokoro Audiobook Studio backend
Run from your audiobook working directory:
    python serve.py           (or via the kokoro-studio launcher)
Then open:  http://localhost:7891
"""

import http.server
import json
import mimetypes
import os
import subprocess
import sys
import threading
import urllib.parse
from pathlib import Path

PORT       = 7891
SCRIPT_DIR = Path(__file__).parent.resolve()   # ~/.local/bin/kokoro-tools/
RUNNER          = SCRIPT_DIR / "kokoro_run.py"
SAMPLER_NATIVE  = SCRIPT_DIR / "voice_sampler.py"
SAMPLER_ACCENTED= SCRIPT_DIR / "voice_sampler_accented_english.py"
VOICES_TXT      = SCRIPT_DIR / "voices.txt"
UI_FILE    = SCRIPT_DIR / "kokoro_ui.html"
WORK_DIR   = Path.cwd()                        # the audiobook project directory

# Voice sample directories live next to this script in kokoro-tools/
SAMPLES_NATIVE   = SCRIPT_DIR / "voice_samples"
SAMPLES_ACCENTED = SCRIPT_DIR / "voice_samples_accented_english"

# HuggingFace cached voice tensors
VOICES_PATH = Path.home() / ".cache/huggingface/hub/models--hexgrad--Kokoro-82M/snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987/voices"

# Custom blended voice output paths
CUSTOM_PT  = { "m": VOICES_PATH / "cm_voice.pt",  "f": VOICES_PATH / "cf_voice.pt" }
CUSTOM_WAV = { "m": SAMPLES_ACCENTED / "cm_voice.wav", "f": SAMPLES_ACCENTED / "cf_voice.wav" }

# Last generated sample audio path (for playback route)
last_sample_path: dict = {"path": None}

# Track running jobs  {job_id: Popen}
jobs: dict[str, subprocess.Popen] = {}
jobs_lock = threading.Lock()


def list_txt_files() -> list[dict]:
    results = []
    for f in sorted(WORK_DIR.iterdir()):
        if not (f.is_file() and f.suffix.lower() == ".txt"):
            continue
        lines = words = 0
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                stripped = line.strip()
                if stripped:
                    lines += 1
                    words += len(stripped.split())
        except Exception:
            pass
        results.append({"name": f.name, "lines": lines, "words": words})
    return results


def list_samples() -> dict:
    """Return which voice IDs have samples in each directory."""
    def wavs(d: Path) -> list[str]:
        if not d.exists():
            return []
        return sorted(p.stem for p in d.iterdir() if p.suffix.lower() == ".wav")

    return {
        "native":   wavs(SAMPLES_NATIVE),
        "accented": wavs(SAMPLES_ACCENTED),
    }


class Handler(http.server.BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass  # silence default access log

    # ── routing ──────────────────────────────────────────────────────────

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path   = parsed.path

        if path in ("/", "/index.html"):
            self._serve_file(UI_FILE, "text/html; charset=utf-8")

        elif path == "/user_guide.html":
            self._serve_file(SCRIPT_DIR / "user_guide.html", "text/html; charset=utf-8")

        elif path == "/api/files":
            self._json(list_txt_files())

        elif path == "/api/samples":
            self._json(list_samples())

        elif path.startswith("/samples/"):
            # /samples/native/af_bella.wav
            # /samples/accented/bf_emma.wav
            parts = path.split("/")   # ['', 'samples', 'native|accented', 'voice.wav']
            if len(parts) == 4:
                bucket, fname = parts[2], parts[3]
                if bucket == "native":
                    wav = SAMPLES_NATIVE / fname
                elif bucket == "accented":
                    wav = SAMPLES_ACCENTED / fname
                else:
                    wav = None
                if wav and wav.exists() and wav.suffix.lower() == ".wav":
                    self._serve_file(wav, "audio/wav")
                    return
            self._text(404, "Sample not found")

        elif path == "/api/generate-samples":
            mode = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("mode", ["native"])[0]
            self._launch_sampler(mode)

        elif path == "/api/sample-audio-file":
            p = last_sample_path.get("path")
            if p and Path(p).exists():
                self._serve_file(Path(p), "audio/wav")
            else:
                self._text(404, "No sample audio generated yet")

        elif path == "/api/blend-status":
            self._json({
                "m": {"pt": CUSTOM_PT["m"].exists(), "wav": CUSTOM_WAV["m"].exists()},
                "f": {"pt": CUSTOM_PT["f"].exists(), "wav": CUSTOM_WAV["f"].exists()},
            })

        elif path == "/api/status":
            qs     = urllib.parse.parse_qs(parsed.query)
            job_id = qs.get("job_id", [None])[0]
            if job_id and job_id in jobs:
                proc = jobs[job_id]
                code = proc.poll()
                self._json({"done": code is not None, "exit_code": code})
            else:
                self._json({"done": True, "exit_code": None})

        else:
            self._text(404, "Not found")

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path

        if path == "/api/run":
            body = self._read_json()
            if body is None:
                return
            self._handle_run(body)

        elif path == "/api/stop":
            body = self._read_json()
            if body:
                job_id = body.get("job_id")
                with jobs_lock:
                    proc = jobs.get(job_id)
                if proc:
                    proc.terminate()
                    self._json({"ok": True})
                else:
                    self._json({"ok": False, "error": "job not found"})

        elif path == "/api/quick-text":
            body = self._read_json()
            if body is None:
                return
            self._handle_quick_text(body)

        elif path == "/api/resolve-dir":
            body = self._read_json()
            if body is None:
                return
            # Best-effort: look for a folder matching the given name near WORK_DIR
            name = (body.get("name") or "").strip()
            candidate = WORK_DIR / name
            if candidate.is_dir():
                self._json({"path": str(candidate)})
            else:
                self._json({"path": None})

        elif path == "/api/sample-audio":
            body = self._read_json()
            if body is None:
                return
            self._handle_sample_audio(body)

        elif path == "/api/blend":
            body = self._read_json()
            if body is None:
                return
            self._handle_blend(body)

        else:
            self._text(404, "Not found")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    # ── handlers ─────────────────────────────────────────────────────────

    def _handle_run(self, body: dict):
        files     = body.get("files", [])
        voice     = body.get("voice", "af_bella")
        speed     = str(body.get("speed", 1.0))
        out       = body.get("output_dir", "./audiobook_output")
        lang      = body.get("lang_code", "a")

        if not files:
            self._json({"error": "no files selected"}, 400)
            return

        if not RUNNER.exists():
            msg = f"kokoro_run.py not found at {RUNNER}"
            print(f"[ERROR] {msg}")
            self._json({"error": msg}, 500)
            return

        cmd = [
            sys.executable, str(RUNNER),
            "--files",    *files,
            "--voice",    voice,
            "--speed",    speed,
            "--output",   out,
            "--lang",     lang,
            "--work-dir", str(WORK_DIR),
        ]

        print(f"[LAUNCH] voice={voice} speed={speed} files={files}")
        print(f"[LAUNCH] cmd: {' '.join(cmd)}")

        try:
            term_cmd, method = _terminal_cmd(cmd)
            print(f"[LAUNCH] terminal method: {method}")
            print(f"[LAUNCH] terminal cmd: {' '.join(str(x) for x in term_cmd)}")
            proc = subprocess.Popen(term_cmd, cwd=str(WORK_DIR))
            job_id = str(id(proc))
            with jobs_lock:
                jobs[job_id] = proc
            print(f"[LAUNCH] started PID {proc.pid}")
            self._json({"job_id": job_id, "started": True, "method": method, "pid": proc.pid})
        except Exception as e:
            print(f"[ERROR] Failed to launch: {e}")
            self._json({"error": str(e), "started": False}, 500)

    def _launch_sampler(self, mode: str):
        if mode == "accented":
            script  = SAMPLER_ACCENTED
            out_dir = str(SAMPLES_ACCENTED)
        else:
            script  = SAMPLER_NATIVE
            out_dir = str(SAMPLES_NATIVE)

        if not script.exists():
            self._json({"ok": False, "error": f"{script.name} not found in {SCRIPT_DIR}"}, 404)
            return
        if not VOICES_TXT.exists():
            self._json({"ok": False, "error": f"voices.txt not found in {SCRIPT_DIR}"}, 404)
            return

        cmd = [sys.executable, str(script), str(VOICES_TXT), out_dir]
        print(f"[SAMPLER] mode={mode}  script={script.name}  output={out_dir}")
        try:
            term_cmd, method = _terminal_cmd(cmd)
            proc = subprocess.Popen(term_cmd, cwd=str(SCRIPT_DIR))
            print(f"[SAMPLER] started PID {proc.pid} via {method}")
            self._json({"ok": True, "mode": mode, "pid": proc.pid})
        except Exception as e:
            print(f"[SAMPLER ERROR] {e}")
            self._json({"ok": False, "error": str(e)}, 500)

    def _handle_quick_text(self, body: dict):
        text   = (body.get("text") or "").strip()
        voice  = body.get("voice", "af_bella")
        speed  = str(body.get("speed", 1.0))
        out    = body.get("output_dir", "./audiobook_output")

        if not text:
            self._json({"ok": False, "error": "No text provided"}, 400)
            return

        # Write text to a temp file so kokoro_run.py can read it normally
        import tempfile
        fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="kokoro_quick_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            self._json({"ok": False, "error": str(e)}, 500)
            return

        # Resolve output so the WAV lands at <out>/quick_text_output.wav
        abs_out = str((WORK_DIR / out).resolve())

        cmd = [
            sys.executable, str(RUNNER),
            "--files",     "quick_text_output.txt",
            "--voice",     voice,
            "--speed",     speed,
            "--output",    abs_out,
            "--lang",      "a",
            "--work-dir",  os.path.dirname(tmp_path),
        ]

        # Rename temp file to match the expected output name
        import shutil
        named_tmp = os.path.join(os.path.dirname(tmp_path), "quick_text_output.txt")
        shutil.move(tmp_path, named_tmp)

        print(f"[QUICK] voice={voice} speed={speed} chars={len(text)} output={abs_out}")
        try:
            term_cmd, method = _terminal_cmd(cmd)
            proc = subprocess.Popen(term_cmd, cwd=os.path.dirname(named_tmp))
            job_id = str(id(proc))
            with jobs_lock:
                jobs[job_id] = proc
            self._json({"started": True, "job_id": job_id, "pid": proc.pid})
        except Exception as e:
            print(f"[QUICK ERROR] {e}")
            self._json({"started": False, "error": str(e)}, 500)

    def _handle_sample_audio(self, body: dict):
        text     = body.get("text", "").strip()
        voice_id = body.get("voice", "af_bella")
        speed    = float(body.get("speed", 1.0))
        out_dir  = body.get("output_dir", "./audiobook_output")
        lang     = body.get("lang_code", "a")

        if not text:
            self._json({"ok": False, "error": "No text provided"}, 400)
            return

        # Resolve output dir relative to WORK_DIR
        out_path = (WORK_DIR / out_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        wav_file = out_path / "sample_audio.wav"

        print(f"[SAMPLE] voice={voice_id} speed={speed} chars={len(text)}")

        # Determine if custom voice — same logic as kokoro_run.py
        CUSTOM_IDS = {"cm_voice", "cf_voice"}
        vp = (Path.home() / ".cache/huggingface/hub"
              / "models--hexgrad--Kokoro-82M"
              / "snapshots/f3ff3571791e39611d31c381e3a41a3af07b4987/voices")

        if voice_id in CUSTOM_IDS:
            pt = vp / f"{voice_id}.pt"
            voice_load = f"""
import torch
voice_arg = torch.load(r"{pt}", weights_only=True)
"""
        else:
            voice_load = f'voice_arg = "{voice_id}"'

        gen_code = (
            "import numpy as np, soundfile as sf\n"
            "from kokoro import KPipeline\n"
            f"{voice_load}\n"
            f"pipeline = KPipeline(lang_code='{lang}')\n"
            f"text = {repr(text)}\n"
            "chunks = []\n"
            f"for gs, ps, audio in pipeline(text, voice=voice_arg, speed={speed}, split_pattern=r'\\n+'):\n"
            "    chunks.append(audio)\n"
            "if chunks:\n"
            f"    sf.write(r\"{wav_file}\", np.concatenate(chunks), 24000)\n"
            "    print('OK')\n"
            "else:\n"
            "    print('NO_AUDIO')\n"
        )

        try:
            result = subprocess.run(
                [sys.executable, "-c", gen_code],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0 or "NO_AUDIO" in result.stdout:
                err = (result.stderr or result.stdout)[-400:]
                print(f"[SAMPLE ERROR] {err}")
                self._json({"ok": False, "error": err}, 500)
            else:
                last_sample_path["path"] = str(wav_file)
                print(f"[SAMPLE OK] {wav_file}")
                self._json({"ok": True, "filename": wav_file.name})
        except subprocess.TimeoutExpired:
            self._json({"ok": False, "error": "Timed out (>120s)"}, 500)
        except Exception as e:
            self._json({"ok": False, "error": str(e)}, 500)

    def _handle_blend(self, body: dict):
        gender = body.get("gender", "f")        # "m" or "f"
        v1_id  = body.get("voice1", "")
        v2_id  = body.get("voice2", "")
        mix1   = float(body.get("mix1", 0.5))   # fraction for voice1 e.g. 0.6
        mix2   = round(1.0 - mix1, 2)

        if not v1_id or not v2_id:
            self._json({"ok": False, "error": "Both voices required"}, 400)
            return
        if v1_id == v2_id:
            self._json({"ok": False, "error": "Select two different voices"}, 400)
            return

        v1_pt = VOICES_PATH / f"{v1_id}.pt"
        v2_pt = VOICES_PATH / f"{v2_id}.pt"
        for pt, name in [(v1_pt, v1_id), (v2_pt, v2_id)]:
            if not pt.exists():
                self._json({"ok": False, "error": f"{name}.pt not found in HF cache at {pt}"}, 404)
                return

        out_pt  = CUSTOM_PT[gender]
        out_wav = CUSTOM_WAV[gender]
        label   = "cm_voice" if gender == "m" else "cf_voice"
        vp      = str(VOICES_PATH)
        ow      = str(out_wav)
        op      = str(out_pt)

        print(f"[BLEND] {v1_id}x{mix1} + {v2_id}x{mix2} -> {out_pt.name}")

        # Run in a subprocess so torch/kokoro are imported fresh in the venv
        blend_code = (
            "import torch, numpy as np, soundfile as sf\n"
            "from pathlib import Path\n"
            "from kokoro import KPipeline\n"
            f"vp = Path(r\"{vp}\")\n"
            f"v1 = torch.load(vp / \"{v1_id}.pt\", weights_only=True)\n"
            f"v2 = torch.load(vp / \"{v2_id}.pt\", weights_only=True)\n"
            f"blended = {mix1} * v1 + {mix2} * v2\n"
            f"torch.save(blended, r\"{op}\")\n"
            "print('PT saved')\n"
            "pipeline = KPipeline(lang_code='a')\n"
            "text = 'Hello, this is my custom narrative voice. "
            "I created this blend to match my own speaking style.'\n"
            "chunks = []\n"
            "for gs, ps, audio in pipeline(text, voice=blended):\n"
            "    chunks.append(audio)\n"
            "combined = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]\n"
            f"sf.write(r\"{ow}\", combined, 24000)\n"
            "print('WAV saved')\n"
        )

        try:
            result = subprocess.run(
                [sys.executable, "-c", blend_code],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                err = (result.stderr or result.stdout)[-600:]
                print(f"[BLEND ERROR]\n{err}")
                self._json({"ok": False, "error": err}, 500)
            else:
                print(f"[BLEND OK] {result.stdout.strip()}")
                self._json({"ok": True, "label": label, "wav": out_wav.name,
                            "v1": v1_id, "v2": v2_id, "mix1": mix1, "mix2": mix2})
        except subprocess.TimeoutExpired:
            self._json({"ok": False, "error": "Blend timed out (>120s)"}, 500)
        except Exception as e:
            self._json({"ok": False, "error": str(e)}, 500)

    # ── helpers ───────────────────────────────────────────────────────────

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, data, code=200):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _text(self, code, msg):
        body = msg.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path, mime: str):
        if not path.exists():
            self._text(404, f"{path.name} not found")
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Accept-Ranges", "bytes")
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            return json.loads(self.rfile.read(length))
        except Exception:
            self._text(400, "Bad JSON")
            return None


def _terminal_cmd(cmd: list[str]) -> tuple[list[str], str]:
    """
    Write a self-contained shell script to /tmp and open it in a terminal.
    Returns (terminal_cmd_list, method_description).
    Raises RuntimeError if no terminal emulator can be found.
    """
    import shutil, platform, tempfile, stat, shlex

    system = platform.system()

    if system == "Linux":
        # Write a temp shell script — avoids ALL quoting / escaping issues
        # Detect and activate a virtualenv if serve.py is running inside one,
        # so the terminal inherits the same Python env (with kokoro installed).
        import sysconfig
        venv_activate = ""
        venv = os.environ.get("VIRTUAL_ENV", "")
        if not venv:
            # sys.executable might be inside a venv even without VIRTUAL_ENV set
            prefix = sysconfig.get_config_var("prefix") or ""
            activate = os.path.join(prefix, "bin", "activate")
            if os.path.isfile(activate):
                venv_activate = f"source {shlex.quote(activate)}"
        elif os.path.isfile(os.path.join(venv, "bin", "activate")):
            venv_activate = f"source {shlex.quote(os.path.join(venv, 'bin', 'activate'))}"

        # Use the exact Python that is running serve.py — guaranteed to have kokoro
        py = shlex.quote(sys.executable)
        run_cmd = " ".join([py] + [shlex.quote(str(c)) for c in cmd[1:]])

        script_lines = [
            "#!/usr/bin/env bash",
            f"cd {shlex.quote(str(WORK_DIR))}",
        ]
        if venv_activate:
            script_lines.append(venv_activate)
        script_lines += [
            run_cmd,
            'echo ""',
            'echo "━━━  job finished  ━━━"',
            'read -p "Press Enter to close…"',
        ]
        fd, script_path = tempfile.mkstemp(suffix=".sh", prefix="kokoro_job_")
        with os.fdopen(fd, 'w') as f:
            f.write("\n".join(script_lines) + "\n")
        os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IEXEC)

        # Try terminals in preference order
        # Each entry: (binary, args_before_script, args_after_script)
        candidates = [
            ("gnome-terminal",   ["--"],               [script_path]),
            ("konsole",          ["-e"],                [script_path]),
            ("xfce4-terminal",   ["-e"],                [script_path]),
            ("tilix",            ["-e"],                [script_path]),
            ("lxterminal",       ["-e"],                [script_path]),
            ("mate-terminal",    ["-e"],                [script_path]),
            ("xterm",            ["-e"],                [script_path]),
            ("x-terminal-emulator", ["-e"],             [script_path]),
        ]
        for binary, pre, post in candidates:
            if shutil.which(binary):
                return ([binary] + pre + post, binary)

        # Last resort: run in the background without a window and log to file
        log = str(WORK_DIR / "kokoro_job.log")
        bg_cmd = f"{' '.join(shlex.quote(str(c)) for c in cmd)} > {shlex.quote(log)} 2>&1"
        print(f"[WARN] No terminal emulator found. Running headless, logging to {log}")
        return (["bash", "-c", bg_cmd], f"headless→{log}")

    elif system == "Darwin":
        import shlex
        escaped = " ".join(shlex.quote(str(c)) for c in cmd)
        apple   = f'tell application "Terminal" to do script "{escaped}"'
        return (["osascript", "-e", apple], "Terminal.app")

    else:
        # Windows
        args = " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd)
        return (["cmd", "/c", "start", "cmd", "/k", args], "cmd.exe")


def main():
    print(f"╔══════════════════════════════════════════════╗")
    print(f"║   Kokoro Audiobook Studio                    ║")
    print(f"║   http://localhost:{PORT}                     ║")
    print(f"║   Tools dir : {str(SCRIPT_DIR)[:30]:<30}║")
    print(f"║   Work dir  : {str(WORK_DIR)[:30]:<30}║")
    print(f"║   Ctrl-C to stop                             ║")
    print(f"╚══════════════════════════════════════════════╝")

    print(f"  Python   : {sys.executable}")
    print(f"  Venv     : {os.environ.get('VIRTUAL_ENV', '(none detected)')}")
    print(f"  Samples (native)  : {SAMPLES_NATIVE}")
    print(f"  Samples (accented): {SAMPLES_ACCENTED}")
    print()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
