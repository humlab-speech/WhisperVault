# WhisperVault

Runs [WhisperX](https://github.com/m-bain/whisperX) — automatic speech recognition with word-level
timestamps, forced alignment, and speaker diarization — inside a fully network-isolated container.

The core guarantee is simple: once models are downloaded, **the transcription process has zero
internet access** (`--network=none` in podman).  Model weights are mounted read-only from the host
filesystem; the container never writes anything to disk.  Communication between the host and the
container happens exclusively through a **Unix Domain Socket (UDS)** — no TCP port is opened by the
transcription container itself.

An optional lightweight **nginx sidecar container** can sit in front of the socket and expose a TCP
port for network access from other machines or services.  This keeps the isolation boundary clean:
the transcription container stays air-gapped while the nginx container handles all networking.

---

## Prerequisites

These commands are executed on the *host* machine (not inside the container).  You
only need a very small Python environment for the management scripts; all heavy
work runs inside the podman container.

- **Python** 3.12 or later with a simple virtualenv.  We keep a host-side venv
  at `./.venv` in the repository to isolate dependencies.
- [`podman`](https://podman.io/) installed and configured.  (Docker works too if
  you prefer; just substitute `docker` for the `podman` commands below.)
- **httpx** — used by both `container/manage.py` and `transcribe.py` for talking
  to the Unix socket.  Install it inside your venv:

```bash
python -m venv .venv          # one-time setup
source .venv/bin/activate     # every shell before using the repo
pip install httpx pre-commit  # httpx is the only runtime dependency; pre-commit is optional
# or simply:
#   pip install -r requirements.txt

```

With the venv active you can run the host utilities like this:

```bash
python container/manage.py start --help     # note the path to the script
python transcribe.py --help
```

`manage.py` lives in `container/` simply to keep the root directory cleaner;
calling it via `python container/manage.py` is intentional and stable.

Once the environment is set up you can follow the quickstart below.

## Quickstart

### 1 — Build the images

```bash
# From the repository root:
podman build -t whisperx-local -f container/Containerfile .
podman build -t whisperx-nginx -f container/nginx/Containerfile container/nginx/
```

### 2 — Start the transcription server

```bash
python container/manage.py start \
    --model /models/extra/kb-whisper-large-ct2 \
    --align-model /models/extra/wav2vec2-large-voxrex-swedish \
    --diarize-model /models/extra/pyannote-speaker-diarization \
    --device cpu \
    --compute-type float32 \
    --language sv
```

The start script watches the podman container state; if the container exits
before the socket becomes ready you’ll see an immediate error and a pointer
to `podman logs` instead of waiting forever.  This avoids silent hangs when a
crash (e.g. GPU driver issue) occurs.  If you request CUDA but it is not
available (drivers missing, `--gpus` not passed, or CTranslate2 falling back
silently) the server will exit with a clear error — it will never pretend to be
on CUDA while actually running on CPU.  You can rerun the same command –
`--replace` will stop any old instance for you.

The command polls until the model is loaded (~15–60 s on first run) and prints a ready confirmation.
The container runs with `--network=none` and `--cap-drop=ALL` — no internet access at any point.

### 3 — (Optional) Start the nginx sidecar

Only needed if you want to reach the API over TCP from another machine or service.
Skip this if you are running everything on the same host.

```bash
# Localhost only (safe default):
python container/manage.py start-nginx

# Exposed to the network on port 8088:
python container/manage.py start-nginx --listen-host 0.0.0.0 --port 8088
```

### 4 — Transcribe

**Using `transcribe.py` (recommended for local use):**

```bash
# Install the only dependency if you don't have it:
pip install httpx

# Transcribe with speaker diarization, write SRT to ./output/:
python transcribe.py audio.wav --language sv --diarize --format srt --output-dir output/

# Multiple formats at once:
python transcribe.py audio.wav --language sv --diarize --format srt txt json --output-dir output/

# Print plain text to stdout (useful in scripts):
python transcribe.py audio.wav --language sv --format txt --print

# Check server status and available models:
python transcribe.py --status
python transcribe.py --models
```

**Using `curl` directly over the socket:**

curl returns the raw JSON API response (including the formatted text inside `outputs`).
Use `transcribe.py` if you want files written to disk automatically.

```bash
# Returns JSON to stdout — pipe through jq to extract a specific format:
curl -s --unix-socket /tmp/whisperx-api/whisperx.sock \
    -X POST http://localhost/transcribe \
    -F "audio=@audio.wav" \
    -F 'params={"language":"sv","diarize":true,"output_format":"srt"}' \
  | jq -r '.outputs.srt' > output/audio.srt
```

**Using `curl` through the nginx sidecar (from any machine on the network):**

```bash
curl -s http://<host-ip>:8088/transcribe \
    -F "audio=@audio.wav" \
    -F 'params={"language":"sv","diarize":true,"output_format":"srt"}' \
  | jq -r '.outputs.srt' > audio.srt
```

### Examples

The commands shown above are the primary curl-based translation of what
`transcribe.py` does automatically.  A minimal shell example that writes
the SRT output to a file is:

```bash
curl -s --unix-socket /tmp/whisperx-api/whisperx.sock \
    -X POST http://localhost/transcribe \
    -F "audio=@audio.wav" \
    -F 'params={"language":"sv","diarize":true,"output_format":"srt"}' \
  | jq -r '.outputs.srt' > audio.srt
```

and the same request through the HTTP sidecar looks like this:

```bash
curl -s http://localhost:8088/transcribe \
    -F "audio=@audio.wav" \
    -F 'params={"language":"sv","diarize":true,"output_format":"srt"}' \
  | jq -r '.outputs.srt' > audio.srt
```

If you prefer Python, the `transcribe.py` client already wraps this logic
and takes care of sending files and writing multiple formats.  In your own
script you can replicate the behaviour using `httpx` as shown in the
`container/client.py` module or simply import and call `container.client.
transcribe()` around whichever network code you need.

### 5 — Switch models without restarting

```bash
# See what's available:
python transcribe.py --models

# Switch the loaded ASR model on the fly:
python container/manage.py reload --model Systran/faster-whisper-small

# Or directly via curl:
curl --unix-socket /tmp/whisperx-api/whisperx.sock \
    -X POST http://localhost/reload \
    -H 'Content-Type: application/json' \
    -d '{"model": "/models/extra/faster-whisper-large-v3-ct2"}'
```

### 6 — Stop

```bash
python container/manage.py stop          # stop transcription container
python container/manage.py stop-nginx    # stop nginx sidecar (if running)
```

---

## Folder structure

```
WhisperVault/
│
├── container/                  Host-side management and container definitions
│   ├── Containerfile           Builds the whisperx-local image (python:3.12-slim base)
│   ├── server.py               FastAPI server that runs *inside* the container
│   │                           Listens on a Unix socket, never on a TCP port
│   ├── manage.py               Host-side CLI: build, start, stop, transcribe, reload
│   ├── client.py               Thin Python client library for programmatic access
│   └── nginx/
│       ├── Containerfile       Builds the whisperx-nginx sidecar image (nginx:alpine)
│       └── nginx.conf          Non-root nginx config: UDS proxy on port 8080
│
├── models/                     Project-local model cache (not tracked in git)
│   ├── kb-whisper-large-ct2/   Swedish ASR model (CTranslate2 format)
│   ├── faster-whisper-large-v3-ct2/  English/multilingual ASR (CTranslate2)
│   ├── wav2vec2-large-voxrex-swedish/  Swedish forced alignment
│   ├── pyannote-speaker-diarization/   Speaker diarization pipeline
│   ├── pyannote-segmentation/          Segmentation backbone
│   ├── paraphrase-multilingual-MiniLM-L12-v2/  Speaker embeddings
│   └── cache/
│       └── torch/hub/
│           └── checkpoints/    English alignment (torchaudio wav2vec2)
│
├── whisperx/                   The whisperx Python package (separate git repo / subdir)
│   └── whisperx/               Package source — installed into the container image
│
├── output/                     Transcription results written by manage.py (not tracked)
├── input/                      Audio files to transcribe (not tracked)
│
└── CLAUDE.md                   Symlink → .github/copilot-instructions.md (agent instructions)
```

> `models/`, `input/`, and `output/` are excluded from git via `.gitignore`.
> `whisperx/` is tracked as a **git submodule**; after cloning you must run:
> `git submodule update --init --recursive`

---

## Required models

All models are stored as **plain directories** under `./models/` — no HuggingFace symlink caches.
This makes the project fully portable: `tar`, `rsync`, `scp`, or any copy method works without
breaking anything.

The `./models/` directory is mounted read-only at `/models/extra` inside the container.
`TORCH_HOME` is set to `/models/extra/cache/torch` so PyTorch finds its hub checkpoints there.

### Model inventory

| Directory | Size | Purpose | Used by |
|---|---|---|---|
| `kb-whisper-large-ct2/` | 2.9 GB | Swedish ASR (CTranslate2 format) | `--model` |
| `faster-whisper-large-v3-ct2/` | 2.9 GB | Multilingual/English ASR (CTranslate2) | `--model` |
| `wav2vec2-large-voxrex-swedish/` | 2.4 GB | Swedish forced alignment | `--align-model` |
| `paraphrase-multilingual-MiniLM-L12-v2/` | 926 MB | Speaker embeddings (all languages) | diarization sub-model |
| `cache/torch/hub/checkpoints/` | 361 MB | English forced alignment (torchaudio) | automatic for `en` |
| `pyannote-speaker-diarization/` | 32 MB | Speaker diarization pipeline | `--diarize-model` |
| `pyannote-segmentation/` | 17 MB | Segmentation backbone | diarization sub-model |

### Downloading models

ASR models use the **CTranslate2** format published by Systran (or KBLab for Swedish).
All other models are downloaded as plain directories via `huggingface_hub.snapshot_download()`.

```bash
# Example: download a CT2 model
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Systran/faster-whisper-large-v3', local_dir='models/faster-whisper-large-v3-ct2',
                  ignore_patterns=['*.md', '*.gitattributes'])
"
```

The pyannote diarization models are **gated** on HuggingFace — you must accept the licence on the
HF website and pass `token=` to `snapshot_download()` (or set `HF_TOKEN`) the first time you
download them.  Once the files are in `models/`, no token is needed for offline inference.

### Using models

Model paths are passed to `manage.py start` as **container-side paths** (prefixed with
`/models/extra/`):

You can also run on GPU.  On a native Linux host with up‑to‑date drivers the
`--gpus all` flag is sufficient; podman will expose the appropriate `/dev/
`nvidia*` devices and libraries.  The `manage.py` helper will also add a
matching `--device nvidia.com/gpu=…` binding automatically when you supply
`--gpus`, which avoids problems with container images that expect the older
syntax.  If you’re using Podman Desktop/Podman Machine you’ll need to install
the NVIDIA Container Toolkit in the VM and generate a CDI spec (`nvidia-ctk cdi generate --output /etc/cdi/nvidia.yaml`).


```bash
# Swedish (CPU):
python container/manage.py start \
    --model /models/extra/kb-whisper-large-ct2 \
    --align-model /models/extra/wav2vec2-large-voxrex-swedish \
    --diarize-model /models/extra/pyannote-speaker-diarization \
    --device cpu --compute-type float32 --language sv

# Swedish on GPU (requires podman with `--gpus` support):
python container/manage.py start \
    --model /models/extra/kb-whisper-large-ct2 \
    --align-model /models/extra/wav2vec2-large-voxrex-swedish \
    --diarize-model /models/extra/pyannote-speaker-diarization \
    --device cuda --gpus all --language sv

# English (alignment model is built into torchaudio, no --align-model needed):
python container/manage.py start \
    --model /models/extra/faster-whisper-large-v3-ct2 \
    --diarize-model /models/extra/pyannote-speaker-diarization \
    --device cpu --compute-type float32 --language en
```

---

## Transferring to another machine

The project is designed to be fully portable.  All models are plain files — no symlinks, no
HuggingFace blob caches.

### What to transfer

| What | In git? | How to get it |
|---|---|---|
| Repository code | ✅ | `git clone` |
| `whisperx/` package source | ✅ | `git clone --recurse-submodules <repo-url>` (or `git submodule update --init --recursive`) |
| `models/` (all model weights) | ❌ | Copy from existing machine **or** re-download (see above) |
| `.venv/` (host-side Python venv) | ❌ | Recreate: `python -m venv .venv && pip install httpx pre-commit` |
| Container images | ❌ | Rebuild: `podman build -t whisperx-local -f container/Containerfile .` |

### Transfer via tar (recommended for air-gapped machines)

```bash
# On the source machine — exclude venv, output, input:
tar cf WhisperVault.tar \
    --exclude='.venv' --exclude='output' --exclude='input' \
    --exclude='models/*/.cache' \
    -C /path/to WhisperVault/

# On the target machine:
tar xf WhisperVault.tar
cd WhisperVault
python -m venv .venv && source .venv/bin/activate && pip install httpx
podman build -t whisperx-local -f container/Containerfile .
python container/manage.py start \
    --model /models/extra/faster-whisper-large-v3-ct2 \
    --device cpu --compute-type float32 --language en
```

### Transfer via git clone + model copy

```bash
# 1. Clone the repo
# (Note: this repo uses a git submodule for whisperx)
git clone --recurse-submodules <repo-url> WhisperVault && cd WhisperVault

# 2. (If you cloned without --recurse-submodules)
git submodule update --init --recursive

# 3. Copy models from an existing machine (rsync, scp, USB drive, etc.)
rsync -a user@source:/path/to/WhisperVault/models/ models/

# 4. Set up the host venv
python -m venv .venv && source .venv/bin/activate && pip install httpx

# 5. Build and start
podman build -t whisperx-local -f container/Containerfile .
python container/manage.py start --model /models/extra/faster-whisper-large-v3-ct2 \
    --device cpu --compute-type float32 --language en
```

> **Note:** The `models/` directory is ~9.5 GB.  If bandwidth is limited, transfer only the models
> you need (e.g. skip the Swedish models if you only need English, or vice versa).

---

## Building the images

Both images must be built from the repository root.

```bash
# Transcription image (built from whisperx source + container/server.py)
podman build -t whisperx-local -f container/Containerfile .

# nginx sidecar image (Alpine-based, only needed if you want TCP access)
podman build -t whisperx-nginx -f container/nginx/Containerfile container/nginx/
```

Or via `manage.py`:

```bash
python container/manage.py build-nginx   # builds only the nginx image
```

---

## Starting the server

### Transcription container

```bash
python container/manage.py start \
    --model /models/extra/kb-whisper-large-ct2 \
    --align-model /models/extra/wav2vec2-large-voxrex-swedish \
    --diarize-model /models/extra/pyannote-speaker-diarization \
    --device cpu \
    --compute-type float32 \
    --language sv
```

This runs the container with:
- `--network=none` — zero network access
- `--cap-drop=ALL` — no Linux capabilities
- `--security-opt=no-new-privileges:true`
- a read-only volume mount (`./models/` → `/models/extra`)
- one read-write bind for the socket directory (`/tmp/whisperx-api` → `/run/api`)

The server loads the ASR model once at startup, then stays resident.  Subsequent requests reuse the
loaded model, so there is no startup overhead per transcription.

The `manage.py start` command polls the socket until the model finishes loading (up to 3 minutes)
and prints a ready confirmation.

### nginx sidecar

```bash
python container/manage.py start-nginx --listen-host 0.0.0.0 --port 8088
```

- `--listen-host 0.0.0.0` — accept connections from any network interface
- `--listen-host 127.0.0.1` (default) — localhost only

The sidecar runs as uid 101 (`nginx` user) with `--cap-drop=ALL`, `--read-only`, and a tmpfs at
`/tmp`.  It has no access to the model weights or audio data — it only proxies HTTP to the socket.

#### Skipping the nginx sidecar

If the host already runs a reverse proxy (nginx, Caddy, Traefik, etc.) you do not need the sidecar.
Instead, point your existing proxy directly at the Unix socket:

```nginx
# In an existing nginx server block:
location /whisperx/ {
    proxy_pass         http://unix:/tmp/whisperx-api/whisperx.sock;
    proxy_read_timeout 3600s;
    client_max_body_size 2G;
}
```

> **Note:** Transcriptions can take a long time for lengthy audio files, so you may need
> to increase your reverse-proxy request timeout (`proxy_read_timeout` in nginx) to avoid
> premature connection closes. `3600s` (60 minutes) is a reasonable starting point.

For Caddy:
```
reverse_proxy unix//tmp/whisperx-api/whisperx.sock
```

---

## Stopping

```bash
python container/manage.py stop          # stop transcription container
python container/manage.py stop-nginx    # stop nginx sidecar
```

---

## Transcribing a file

Via `manage.py` (writes output files to `./output/` by default):

```bash
python container/manage.py transcribe /path/to/audio.wav \
    --language sv \
    --diarize \
    --output-format srt \
    --output-dir ./output
```

Via `curl` (returns JSON — use `jq` to extract the formatted text):

```bash
# Over the Unix socket:
curl -s --unix-socket /tmp/whisperx-api/whisperx.sock \
    -X POST http://localhost/transcribe \
    -F "audio=@/path/to/audio.wav" \
    -F 'params={"language":"sv","diarize":true,"output_format":["srt","txt"]}' \
  | jq -r '.outputs.srt' > output/audio.srt

# Through the nginx sidecar:
curl -s http://localhost:8088/transcribe \
    -F "audio=@/path/to/audio.wav" \
    -F 'params={"language":"sv","diarize":true,"output_format":"txt"}' \
  | jq -r '.outputs.txt' > output/audio.txt
```

---

## HTTP API reference

This section contains enough detail for a client implementation in any language.

### `GET /health`

Returns the current server state.  Useful for liveness/readiness checks.

**Response** `200 OK`:
```json
{
  "ready": true,
  "reloading": false,
  "model": "/models/extra/kb-whisper-large-ct2",
  "device": "cpu",
  "compute_type": "float32",
  "language": "sv",
  "vad_method": "pyannote",
  "align_models_cached": ["sv"],
  "diarize_pipelines_cached": ["/models/extra/pyannote-speaker-diarization"]
}
```

---

### `GET /models`

Scans the model caches that are mounted into the container – both the
HuggingFace hub cache (`/models/hf`) **and** the project-local model
directory (`/models/extra`).  Any subdirectory or HF cache entry is
reported, and each is classified by its role in the whisperx pipeline.

This makes the API useful for discovering models that were downloaded via
`huggingface_hub` as well as manually copied CT2 models and other plain
directories.

```bash
curl --unix-socket /tmp/whisperx-api/whisperx.sock http://localhost/models
# or via nginx:
curl http://localhost:8088/models
```

**Response** `200 OK` (abbreviated):
```json
{
  "available": [
    {
      "model_id": "/models/extra/kb-whisper-large-ct2",
      "role": "asr",
      "loaded": true,
      "description": "Swedish-optimised Whisper large model ...",
      "languages": ["sv"],
      "architecture": "faster-whisper"
    },
    {
      "model_id": "/models/extra/pyannote-speaker-diarization",
      "role": "diarization",
      "loaded": false
    }
  ],
  "by_role": {
    "asr":         [ ... ],
    "alignment":   [ ... ],
    "diarization": [ ... ],
    "vad":         [ ... ],
    "embedding":   [ ... ]
  },
  "currently_loaded": {
    "asr": "/models/extra/kb-whisper-large-ct2",
    "alignment": ["sv"],
    "diarization": []
  }
}
```

**Roles:**

| Role | Meaning |
|---|---|
| `asr` | Can be passed as `model` to `POST /reload` or `manage.py start --model` |
| `alignment` | Forced-alignment model; selected automatically by detected language |
| `diarization` | Speaker diarization pipeline; selectable via `diarize_model` in `/transcribe` |
| `vad` | VAD backbone used internally when `vad_method=pyannote` |
| `embedding` | Speaker embedding model used internally by the diarization pipeline |
| `unknown` | Present in the cache but not in the server's known-model registry |

To switch to a different ASR model that is listed under `role=asr`, use `POST /reload`:
```bash
curl --unix-socket /tmp/whisperx-api/whisperx.sock \
    -X POST http://localhost/reload \
    -H 'Content-Type: application/json' \
    -d '{"model": "openai/whisper-large-v3"}'
```

---

### `GET /params`

Returns the complete parameter schema for both `/transcribe` and `/reload` as a machine-readable
JSON document.  Useful for clients and automated agents that need to discover the API surface
without consulting external documentation.

```bash
curl --unix-socket /tmp/whisperx-api/whisperx.sock http://localhost/params
# or via nginx:
curl http://localhost:8088/params
```

**Response** `200 OK` (abbreviated):
```json
{
  "transcribe_params": {
    "language":   { "type": "string | null", "default": null, "description": "..." },
    "diarize":    { "type": "bool",          "default": false, "description": "..." },
    "output_format": { "type": "string | list[string]", "default": "all",
                       "enum": ["all","txt","srt","vtt","tsv","json","aud"],
                       "description": "..." }
  },
  "reload_params": {
    "model":      { "type": "str", "default": "small", "current_value": "/models/extra/kb-whisper-large-ct2",
                    "env_var": "WHISPERX_MODEL", "description": "..." },
    "beam_size":  { "type": "int", "default": 5,       "current_value": 5,
                    "env_var": "WHISPERX_BEAM_SIZE", "description": "..." },
    "repetition_penalty": { "type": "float", "default": 1.0, "current_value": 1.0,
                    "env_var": "WHISPERX_REPETITION_PENALTY", "description": "Decoding repetition penalty." }
  },
  "output_formats": ["txt", "srt", "vtt", "tsv", "json", "aud"],
  "notes": {
    "transcribe_params_usage": "Send as a JSON-encoded string in the 'params' multipart field ...",
    "reload_params_usage":     "Send as a JSON object body to POST /reload ...",
    "overlap":                 "Some fields exist in both sets; the reload version sets the "
                               "server-wide default, the transcribe version overrides for one request."
  }
}
```

`reload_params` entries always include a `current_value` field showing what is actually loaded
right now, which differs from `default` when the server was started with non-default flags.

---

### `POST /transcribe`

Multipart form upload.  The two fields are:

| Field | Type | Description |
|---|---|---|
| `audio` | binary file | Audio in any format ffmpeg understands (wav, mp3, m4a, flac, ogg, …) |
| `params` | JSON string | Optional per-request parameters (see below) |

**`params` fields** (all optional):

| Key | Type | Default | Description |
|---|---|---|---|
| `language` | string / null | server default | ISO-639-1 code, e.g. `"sv"`, `"en"`.  `null` = auto-detect |
| `task` | `"transcribe"` \| `"translate"` | `"transcribe"` | `"translate"` produces English output regardless of source language |
| `batch_size` | int | 8 | Inference batch size |
| `chunk_size` | int | 30 | VAD chunk size in seconds |
| `no_align` | bool | false | Skip forced alignment (word-level timestamps will be absent) |
| `align_model` | string | auto | Override the alignment model (HF path) |
| `interpolate_method` | `"nearest"` \| `"linear"` \| `"ignore"` | `"nearest"` | How to fill gaps in word-level alignment |
| `return_char_alignments` | bool | false | Include character-level timestamps in output |
| `diarize` | bool | false | Run speaker diarization and annotate segments with speaker labels |
| `min_speakers` | int / null | null | Hint for diarization (minimum expected speakers) |
| `max_speakers` | int / null | null | Hint for diarization (maximum expected speakers) |
| `diarize_model` | string | `"pyannote/speaker-diarization-community-1"` | Override diarization model |
| `speaker_embeddings` | bool | false | Return per-speaker embedding vectors in segments |
| `output_format` | string / list | `"all"` | One of `"txt"`, `"srt"`, `"vtt"`, `"tsv"`, `"json"`, `"aud"`, `"all"`, or a list such as `["srt","txt"]` |
| `highlight_words` | bool | false | Underline each word as it is spoken (SRT/VTT) |
| `max_line_width` | int / null | null | Wrap subtitle lines at this character width |
| `max_line_count` | int / null | null | Maximum lines per subtitle block |
| `verbose` | bool | false | Log detailed progress inside the container |
| `print_progress` | bool | false | Log segment-level progress inside the container |

**Response** `200 OK`:
```json
{
  "language": "sv",
  "duration_seconds": 146.3,
  "segments": [
    {
      "start": 0.0,
      "end": 3.2,
      "text": " [example transcribed text]",
      "words": [
        { "word": "Om",        "start": 0.08, "end": 0.32, "score": 0.95 },
        { "word": "man",       "start": 0.36, "end": 0.52, "score": 0.99 },
        { "word": "vill",      "start": 0.56, "end": 0.76, "score": 0.98 }
      ],
      "speaker": "SPEAKER_01"
    }
  ],
  "outputs": {
    "txt": "[example transcribed text]\n...",
    "srt": "1\n00:00:00,080 --> 00:00:03,200\n[example transcribed text]\n\n..."
  }
}
```

- `segments` is a flat list; each entry has `start` / `end` (seconds, float), `text` (string with
  leading space), and optionally `words` (array) and `speaker` (string, only when `diarize=true`).
- `outputs` contains one key per requested format; the values are the full file contents as strings.
- When `diarize=false` the `speaker` field is absent from segments.
- When `no_align=true` the `words` array is absent from segments.

---

### `POST /reload`

Hot-swap the loaded ASR model without restarting the container.  Send a JSON object containing only
the fields you want to change; unspecified fields keep their current value.

Reload is only needed for parameters that are baked in at model-load time: `model`, `device`,
`compute_type`, `beam_size`, `vad_method`, `vad_onset`, `vad_offset`, etc.  Per-request parameters
(`language`, `diarize`, `output_format`, …) never require a reload.

```bash
curl --unix-socket /tmp/whisperx-api/whisperx.sock \
    -X POST http://localhost/reload \
    -H 'Content-Type: application/json' \
    -d '{"model": "large-v2", "beam_size": 10}'
```

**Response** `200 OK`:
```json
{ "status": "reloaded", "model": "large-v2", "device": "cpu", "compute_type": "float32" }
```

Returns `503` if a reload is already in progress, `500` if loading fails.

---

## Unix Domain Socket — notes for rootless podman

The socket lives at `/tmp/whisperx-api/whisperx.sock` on the host.  Because the container is run
rootless (no root required), it is owned by the current user.

**Socket permissions**

The server explicitly sets `umask(0)` before uvicorn creates the socket, so the socket file is
created with mode `0666` (readable and writable by everyone).  This is necessary because the nginx
sidecar runs as uid 101 (`nginx`), which is different from the user who started the whisperx
container.  Without `0666`, `connect()` would fail with `EACCES`.

```
srw-rw-rw- 1 tomas tomas 0 ... /tmp/whisperx-api/whisperx.sock
```

**Who can connect**

Any process running as the same OS user, or any process with write permission on the socket file,
can connect.  Because the socket is `0666`, any local process can use it — including:

- `curl --unix-socket /tmp/whisperx-api/whisperx.sock …`
- A Node.js / Python / Go service using the `httpx` / `http.Client` / `axios` unix socket option
- Another container with the socket directory bind-mounted (`:z` label for SELinux)

**Accessing the socket from another container**

```bash
podman run --rm \
    -v /tmp/whisperx-api:/run/api:z \
    some-image \
    curl --unix-socket /run/api/whisperx.sock http://localhost/health
```

**Direct socket access from Node.js (example)**

```js
import http from 'http';

const options = {
  socketPath: '/tmp/whisperx-api/whisperx.sock',
  path: '/health',
  method: 'GET',
};
const req = http.request(options, res => {
  let body = '';
  res.on('data', chunk => body += chunk);
  res.on('end', () => console.log(JSON.parse(body)));
});
req.end();
```

For `POST /transcribe` from Node.js, use a multipart form library (e.g. `form-data`) to send the
`audio` file and a `params` JSON string field, targeting the same `socketPath`.

---

## Reloading the model

The running server can switch to a different model without restarting the container:

```bash
python container/manage.py reload --model large-v2 --beam-size 10
```

All cached alignment and diarization pipelines are also invalidated and will be lazily re-loaded on
the next request that needs them.

---

## Security summary

| Container | Network | Capabilities | Root fs |
|---|---|---|---|
| `whisperx-server` | `none` (no network at all) | `--cap-drop=ALL` | writable (Python process needs `/tmp`) |
| `whisperx-nginx` | host TCP port | `--cap-drop=ALL` | read-only + tmpfs at `/tmp` |

Neither container runs as root.  Neither container can write to the host filesystem (model mounts
are `:ro`; the only shared writable path is the socket directory, which contains only the socket
file itself).
