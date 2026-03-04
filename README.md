# WhisperX Offline Annotation

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
    --model KBLab/kb-whisper-large \
    --device cpu \
    --compute-type float32 \
    --language sv
```

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

```bash
curl --unix-socket /tmp/whisperx-api/whisperx.sock \
    -X POST http://localhost/transcribe \
    -F "audio=@audio.wav" \
    -F 'params={"language":"sv","diarize":true,"output_format":"srt"}'
```

**Using `curl` through the nginx sidecar (from any machine on the network):**

```bash
curl http://<host-ip>:8088/transcribe \
    -F "audio=@audio.wav" \
    -F 'params={"language":"sv","diarize":true,"output_format":"srt"}'
```

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
    -d '{"model": "KBLab/kb-whisper-large"}'
```

### 6 — Stop

```bash
python container/manage.py stop          # stop transcription container
python container/manage.py stop-nginx    # stop nginx sidecar (if running)
```

---

## Folder structure

```
WhisperXAnnoteOffline/
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
│   ├── cache/
│   │   └── torch/hub/
│   │       └── checkpoints/    Torch hub checkpoints (wav2vec2 alignment model)
│   ├── kb_whisper_large/       (legacy local copy, superseded by HF hub cache)
│   ├── pyannote_speaker_diarization/
│   │   └── plda/               PLDA scoring weights for diarization
│   └── wav2vec2_large_sv_lm/   Swedish wav2vec2 language model
│
├── whisperx/                   The whisperx Python package (separate git repo / subdir)
│   └── whisperx/               Package source — installed into the container image
│
├── output/                     Transcription results written by manage.py (not tracked)
├── input/                      Audio files to transcribe (not tracked)
│
├── run_whisper_offline.py      Legacy one-shot wrapper (still works, uses standalone mode)
└── README.upstream.md          Original upstream whisperx README for reference
```

> `models/`, `input/`, `output/`, and `whisperx/` are excluded from git via `.gitignore`.

---

## Required models

Two separate model caches are mounted into the container at runtime, both read-only.

### 1 — HuggingFace hub cache  (`~/.cache/huggingface/hub`)

This is the standard HF cache on the host.  The following entries are used:

| Entry | Purpose |
|---|---|
| `models--KBLab--kb-whisper-large` | Main ASR model (Swedish-optimised Whisper large) |
| `models--KBLab--wav2vec2-large-voxrex-swedish` | Forced alignment model for Swedish |
| `models--pyannote--segmentation` | VAD / segmentation backbone |
| `models--pyannote--speaker-diarization-community-1` | Speaker diarization pipeline |
| `models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2` | Speaker embedding model |

The pyannote diarization models are gated on HuggingFace.  You must accept the licence on the HF
website and provide `HF_TOKEN` the first time you download them.  Once cached, the token is not
needed for offline inference.

Inside the container this cache is mounted at `/models/hf` and the env vars `HF_HOME` and
`HF_HUB_CACHE` are both set to that path.  `HF_HUB_OFFLINE=1` is set by default so any attempt
to make a network request will raise an error immediately rather than hanging.

### 2 — Project models directory  (`./models/`)

Mounted at `/models/extra` inside the container.  `TORCH_HOME` is set to `/models/extra/cache/torch`
so PyTorch finds its hub checkpoints here.

The key file is:
```
models/cache/torch/hub/checkpoints/wav2vec2_fairseq_base_ls960_asr_ls960.pth
```
This is the wav2vec2 checkpoint used by whisperx's alignment step.

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
    --model KBLab/kb-whisper-large \
    --device cpu \
    --compute-type float32 \
    --language sv
```

This runs the container with:
- `--network=none` — zero network access
- `--cap-drop=ALL` — no Linux capabilities
- `--security-opt=no-new-privileges:true`
- two read-only volume mounts (HF cache and `./models/`)
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

Via `curl` (through the nginx sidecar on port 8088):

```bash
curl http://localhost:8088/transcribe \
    -F "audio=@/path/to/audio.wav" \
    -F 'params={"language":"sv","diarize":true,"output_format":"srt"}'
```

Via `curl` (directly over the Unix socket, no nginx needed):

```bash
curl --unix-socket /tmp/whisperx-api/whisperx.sock \
    -X POST http://localhost/transcribe \
    -F "audio=@/path/to/audio.wav" \
    -F 'params={"language":"sv","diarize":true,"output_format":"txt"}'
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
  "model": "KBLab/kb-whisper-large",
  "device": "cpu",
  "compute_type": "float32",
  "language": "sv",
  "vad_method": "pyannote",
  "align_models_cached": ["sv"],
  "diarize_pipelines_cached": ["pyannote/speaker-diarization-community-1"]
}
```

---

### `GET /models`

Scans the HuggingFace hub cache (`/models/hf` inside the container) and returns every model found,
classified by its role in the whisperx pipeline.

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
      "model_id": "KBLab/kb-whisper-large",
      "cache_path": "/models/hf/models--KBLab--kb-whisper-large",
      "role": "asr",
      "loaded": true,
      "description": "Swedish-optimised Whisper large model ...",
      "languages": ["sv"],
      "architecture": "faster-whisper"
    },
    {
      "model_id": "pyannote/speaker-diarization-community-1",
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
    "asr": "KBLab/kb-whisper-large",
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
    "model":      { "type": "str", "default": "small", "current_value": "KBLab/kb-whisper-large",
                    "env_var": "WHISPERX_MODEL", "description": "..." },
    "beam_size":  { "type": "int", "default": 5,       "current_value": 5,
                    "env_var": "WHISPERX_BEAM_SIZE", "description": "..." }
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

## One-shot legacy mode

The original workflow (one container invocation per file, then `--rm`) still works via
`run_whisper_offline.py`:

```bash
python run_whisper_offline.py /path/to/audio.wav \
    --model KBLab/kb-whisper-large --device cpu --compute_type float32 \
    --diarize --output_dir ./output --output_format txt
```

This prepends `"standalone"` to the container arguments, causing `server.py` to hand control
directly to the whisperx CLI instead of starting the API server.

---

## Security summary

| Container | Network | Capabilities | Root fs |
|---|---|---|---|
| `whisperx-server` | `none` (no network at all) | `--cap-drop=ALL` | writable (Python process needs `/tmp`) |
| `whisperx-nginx` | host TCP port | `--cap-drop=ALL` | read-only + tmpfs at `/tmp` |

Neither container runs as root.  Neither container can write to the host filesystem (model mounts
are `:ro`; the only shared writable path is the socket directory, which contains only the socket
file itself).
