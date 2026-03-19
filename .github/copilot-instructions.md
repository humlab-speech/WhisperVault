# Project instructions (Copilot + Claude)

This file is read automatically by GitHub Copilot in every session.
CLAUDE.md at the project root is a symlink to this file so Claude Code reads it too.
Keep it up to date when the architecture or conventions change.

---

## What this project is

**WhisperVault** — runs the [whisperx](https://github.com/m-bain/whisperX)
speech-recognition / diarization stack inside a fully network-isolated podman container.
Communication between the host and the container is exclusively over a **Unix Domain Socket (UDS)**.
The transcription container is always started with `--network=none` and `--cap-drop=ALL`.
An optional **nginx sidecar** exposes a TCP port for remote access while keeping the transcription
container air-gapped.

---

## Repository layout

```
container/
  Containerfile          builds whisperx-local (python:3.12-slim)
  server.py              FastAPI app that runs *inside* the container
                           GET  /health   GET  /models   GET  /params
                           POST /transcribe               POST /reload
  manage.py              host-side CLI (start/stop/status/reload/transcribe/
                           build-nginx/start-nginx/stop-nginx)
  client.py              thin httpx client library
  nginx/
    Containerfile        builds whisperx-nginx (nginx:alpine)
    nginx.conf           non-root UDS proxy on port 8080

transcribe.py            standalone CLI client (only dep: httpx)

models/                  project-local model cache  [git-ignored]
output/                  transcription output files [git-ignored]
input/                   audio input files          [git-ignored]
whisperx/                whisperx Python package    [git submodule — pinned commit]
```

---

## Key conventions

### Language / style
- Python 3.12, no type stubs required but use type hints in new code
- `dataclasses` for config (`ModelConfig` in server.py)
- `httpx` (not `requests`) for all HTTP over UDS on the host side
- Async FastAPI handlers; the asyncio lock `_lock` serialises model access

### Pre-commit
- A host-side venv lives at project root (`./.venv`) — activate it before committing:
  ```bash
  source .venv/bin/activate
  ```
- Agents (including Copilot) must always call `configure_python_environment` at the start of a session or before any Python-related operation; this ensures the correct venv is used.
- If `pre-commit` is not yet installed in that venv:
  ```bash
  pip install pre-commit
  pre-commit install
  ```
- Always run `pre-commit run --all-files` (or let the git hook run it) before committing.

### Git
- Always use plain `git` CLI commands in the terminal — never use MCP git tools (e.g. `mcp_gitkraken_*`).
- Stage and commit via `git add` / `git commit`; check status with `git status` / `git diff`.

### Commit messages
- Keep messages short and to the point — one concise subject line describing *what* changed and *why*.
- Do **not** include line counts, file counts, or before/after comparisons.
- Examples of good messages: `fix: correct socket umask before uvicorn start`, `feat: add /models endpoint with HF cache scan`
- Examples of bad messages: *(adds 42 lines to server.py, removes 3 from manage.py)*

### Socket
- Host path: `/tmp/whisperx-api/whisperx.sock` (default, `$WHISPERX_SOCKET`)
- Container path: `/run/api/whisperx.sock`
- The server calls `os.umask(0o000)` before uvicorn creates the socket so
  the file is `0666` — required for the nginx user (uid 101) to `connect()`

### Container security
- whisperx container: `--network=none --cap-drop=ALL --security-opt=no-new-privileges:true`
- nginx container: same caps + `--read-only --tmpfs /tmp:mode=1777`
- Model mounts are always `:ro`
- Only the socket directory is writable (and contains only the socket file)

### Model storage
- All models live as **plain directories** under `./models/` — no HF symlink caches, no hub cache format
- `./models/` → mounted read-only at `/models/extra` inside the container (the **only** model mount)
  - `TORCH_HOME=/models/extra/cache/torch`
  - `HF_HUB_OFFLINE=1` is set so any accidental network request fails fast
  - There is **no** `/models/hf` mount — do not add one
- Each model directory optionally contains an `alias` file with short names and metadata comments:
  - `# role: asr|alignment|diarization|vad|embedding`
  - `# language: sv` (repeatable)
  - `# description: …`
  - Plain-text aliases (one per line) that `manage.py` resolves at call time
- ASR models use **CTranslate2** format (e.g. `models/kb-whisper-large-ct2/`)
- Alignment / diarization models are plain HF snapshot directories
- Container-side paths are passed via `--model`, `--align-model`, `--diarize-model`
  (all prefixed `/models/extra/`)

### server.py internals
- `ModelConfig` dataclass — all fields settable via env vars; see header docstring
- `_TRANSCRIBE_PARAMS` dict — static schema for /transcribe per-request params
- `_read_model_metadata(path)` — reads `# role:` / `# language:` / `# description:` from a model's `alias` file
- `_scan_extra_models()` — scans `/models/extra` at request time for `/models` endpoint
- `_model_config_schema()` — introspects `ModelConfig` at request time for `/params`
- Two categories of params:
  - **reload params** (`ModelConfig` fields): baked in at load time, need `POST /reload`
  - **transcribe params**: evaluated per request, no reload needed
- Some fields (`language`, `batch_size`, `chunk_size`) exist in both — reload sets the
  server default, transcribe overrides for one request only
- **Idle unload**: `_idle_watcher` background task unloads models after `idle_timeout` seconds
  of inactivity (default 120s, `WHISPERX_IDLE_TIMEOUT_SECONDS`, 0 = disabled).  The next
  `/transcribe` auto-reloads from the saved `_config` transparently before proceeding.
  `/health` exposes `idle_unloaded` and `idle_timeout_seconds`.

---

## Common commands

### Build
```bash
podman build -t whisperx-local -f container/Containerfile .
podman build -t whisperx-nginx -f container/nginx/Containerfile container/nginx/
```

### Start / stop
```bash
# GPU notes
# On a real Linux host with NVIDIA/AMD drivers the --gpus flag is enough; podman
# will expose the necessary devices.  In Podman Desktop/Podman Machine you must
# also install the NVIDIA Container Toolkit in the VM and run
# `nvidia-ctk cdi generate --output /etc/cdi/nvidia.yaml` before starting the
# container.  A driver/runtime mismatch will cause a warning and fallback to CPU.

# Swedish (with alignment + diarization)
python container/manage.py start \
    --model /models/extra/kb-whisper-large-ct2 \
    --align-model /models/extra/wav2vec2-large-voxrex-swedish \
    --diarize-model /models/extra/pyannote-speaker-diarization \
    --device cpu --compute-type float32 --language sv

# the start command watches the container state; if the podman process exits before
# the socket appears you'll get an immediate error and a hint to `podman logs`,
# avoiding an infinite hang when the container crashes (e.g. due to a GPU issue).
# If CUDA is requested but unavailable (torch.cuda.is_available() False, or
# CTranslate2 falls back silently) the server raises immediately — it never
# pretends to be on CUDA while running on CPU.

# Same but on GPU (requires podman with --gpus support)
# On a native Linux host with proper NVIDIA drivers the flag is all you need.
# If you're running inside Podman Desktop/Podman Machine you must also install
# the NVIDIA Container Toolkit and generate a CDI spec (`nvidia-ctk cdi generate
# --output /etc/cdi/nvidia.yaml`).  Then either `--gpus all` or
# `--device nvidia.com/gpu=all` will work.
python container/manage.py start \
    --model /models/extra/kb-whisper-large-ct2 \
    --align-model /models/extra/wav2vec2-large-voxrex-swedish \
    --diarize-model /models/extra/pyannote-speaker-diarization \
    --device cuda --gpus all --language sv

# English (torchaudio handles alignment automatically)
python container/manage.py start \
    --model /models/extra/faster-whisper-large-v3-ct2 \
    --diarize-model /models/extra/pyannote-speaker-diarization \
    --device cpu --compute-type float32 --language en

python container/manage.py start-nginx --listen-host 0.0.0.0 --port 8088
python container/manage.py stop
python container/manage.py stop-nginx
```

### Transcribe (CLI)
```bash
python transcribe.py --status
python transcribe.py --models
python transcribe.py audio.wav --language sv --diarize --format srt --output-dir output/
python transcribe.py audio.wav --language sv --format txt --print
```

### Transcribe (curl over socket)
```bash
curl --unix-socket /tmp/whisperx-api/whisperx.sock \
    -X POST http://localhost/transcribe \
    -F "audio=@audio.wav" \
    -F 'params={"language":"sv","diarize":true,"output_format":"srt"}'
```

### Reload model without restart
```bash
python container/manage.py reload --model Systran/faster-whisper-small
# or
curl --unix-socket /tmp/whisperx-api/whisperx.sock \
    -X POST http://localhost/reload \
    -H 'Content-Type: application/json' \
    -d '{"model": "KBLab/kb-whisper-large"}'
```

### Check API surface at runtime
```bash
curl --unix-socket /tmp/whisperx-api/whisperx.sock http://localhost/params
curl --unix-socket /tmp/whisperx-api/whisperx.sock http://localhost/models
```

---

## Available models (as of last session)
- ASR (Swedish): `models/kb-whisper-large-ct2/` — KBLab Swedish Whisper large, CTranslate2 format
- ASR (English): `models/faster-whisper-large-v3-ct2/` — Whisper large-v3, CTranslate2 format
- Alignment (Swedish): `models/wav2vec2-large-voxrex-swedish/` — plain directory
- Alignment (English): built into torchaudio (`wav2vec2_fairseq_base_ls960_asr_ls960.pth` in `models/cache/torch/`)
- Diarization: `models/pyannote-speaker-diarization/` + `models/pyannote-segmentation/` + `models/paraphrase-multilingual-MiniLM-L12-v2/`
- Device: `cpu`, compute_type: `float32`

---

## What NOT to do
- Do not add TCP port mappings (`-p`) to the whisperx container — socket only
- Do not change `--network=none` on the whisperx container
- Do not write to `/models/extra` from inside the container (`:ro`)
- Do not use HuggingFace symlink caches — all models must be plain directories (portability)
- Do not add model weights or audio files to git (`.gitignore` covers `models/`, `input/`, `output/`)
- Do not use `requests` — use `httpx` with `HTTPTransport(uds=...)`
- Do not store secrets in code — `HF_TOKEN` goes in env only

---

## Testing a change
For quick iteration while editing `container/server.py` or `container/manage.py`, start the container in **dev mode** so the host scripts are mounted straight into the container (no rebuild required):
```bash
python container/manage.py start --dev \
    --model /models/extra/kb-whisper-large-ct2 --device cpu --compute-type float32 --language sv
```You can also start without a model (`--model` is optional) and load one later via `POST /reload`.
Once the change works, rebuild the image and start without `--dev` to ensure you’re running the baked-in code:
```bash
podman build -t whisperx-local -f container/Containerfile . && \
python container/manage.py start \
    --model /models/extra/kb-whisper-large-ct2 --device cpu --compute-type float32 --language sv
```
Smoke test:
```bash
curl -s --unix-socket /tmp/whisperx-api/whisperx.sock http://localhost/health | python3 -m json.tool
python transcribe.py --status
```
