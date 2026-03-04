# CLAUDE.md — project context for Claude Code / claude CLI

This file is read automatically by Claude at the start of every session.
Keep it up to date when the architecture or conventions change.
The full user-facing documentation is in README.md.

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
run_whisper_offline.py   legacy one-shot wrapper (still works)

models/                  project-local model cache  [git-ignored]
output/                  transcription output files [git-ignored]
input/                   audio input files          [git-ignored]
whisperx/                whisperx Python package    [git-ignored, separate git repo]
```

---

## Key conventions

### Language / style
- Python 3.12, no type stubs required but use type hints in new code
- `dataclasses` for config (`ModelConfig` in server.py)
- `httpx` (not `requests`) for all HTTP over UDS on the host side
- Async FastAPI handlers; the asyncio lock `_lock` serialises model access

### Pre-commit
- A host-side venv lives at `whisperx/.venv` — activate it before committing:
  ```bash
  source whisperx/.venv/bin/activate
  ```
- If `pre-commit` is not yet installed in that venv:
  ```bash
  pip install pre-commit
  pre-commit install
  ```
- Always run `pre-commit run --all-files` (or let the git hook run it) before committing.

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

### Model caches (two separate locations)
- HF hub: `~/.cache/huggingface/hub` → mounted at `/models/hf` inside container
  - `HF_HOME=/models/hf`  `HF_HUB_CACHE=/models/hf`  `HF_HUB_OFFLINE=1`
- Project models: `./models/` → mounted at `/models/extra`
  - `TORCH_HOME=/models/extra/cache/torch`

### server.py internals
- `ModelConfig` dataclass — all fields settable via env vars; see header docstring
- `_TRANSCRIBE_PARAMS` dict — static schema for /transcribe per-request params
- `_MODEL_METADATA` dict — known model descriptions keyed by HF model ID
- `_scan_hf_cache()` — scans `/models/hf` at request time for `/models` endpoint
- `_model_config_schema()` — introspects `ModelConfig` at request time for `/params`
- Two categories of params:
  - **reload params** (`ModelConfig` fields): baked in at load time, need `POST /reload`
  - **transcribe params**: evaluated per request, no reload needed
- Some fields (`language`, `batch_size`, `chunk_size`) exist in both — reload sets the
  server default, transcribe overrides for one request only

---

## Common commands

### Build
```bash
podman build -t whisperx-local -f container/Containerfile .
podman build -t whisperx-nginx -f container/nginx/Containerfile container/nginx/
```

### Start / stop
```bash
python container/manage.py start \
    --model KBLab/kb-whisper-large --device cpu --compute-type float32 --language sv

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

## Active model (as of last session)
- ASR: `KBLab/kb-whisper-large` (Swedish-optimised Whisper large, KBLab)
- Device: `cpu`, compute_type: `float32`, language: `sv`
- Diarization: `pyannote/speaker-diarization-community-1` (lazy-loaded on first diarize request)
- Alignment: `KBLab/wav2vec2-large-voxrex-swedish` (lazy-loaded on first align request)

---

## What NOT to do
- Do not add TCP port mappings (`-p`) to the whisperx container — socket only
- Do not change `--network=none` on the whisperx container
- Do not write to `/models/hf` or `/models/extra` from inside the container (both `:ro`)
- Do not add model weights or audio files to git (`.gitignore` covers `models/`, `input/`, `output/`)
- Do not use `requests` — use `httpx` with `HTTPTransport(uds=...)`
- Do not store secrets in code — `HF_TOKEN` goes in env only

---

## Testing a change
After editing `server.py` or `container/Containerfile`, rebuild and restart:
```bash
podman build -t whisperx-local -f container/Containerfile . && \
python container/manage.py start \
    --model KBLab/kb-whisper-large --device cpu --compute-type float32 --language sv
```
Smoke test:
```bash
curl -s --unix-socket /tmp/whisperx-api/whisperx.sock http://localhost/health | python3 -m json.tool
python transcribe.py --status
```
