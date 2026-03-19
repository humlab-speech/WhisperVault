#!/usr/bin/env python3
"""
Host-side manager for the whisperx-server container.

The container runs with --network=none.  All communication happens via a
Unix Domain Socket (UDS) that is bind-mounted from the host:

  host path: $SOCKET_DIR/whisperx.sock   (default: /tmp/whisperx-api)
  inside container: /run/api/whisperx.sock

Subcommands
-----------
  start       Build and run the container (loads model, stays resident)
  stop        Stop and remove the container
  status      Print health / current model info
  reload      Ask the running server to reload with new model config
  transcribe  Transcribe an audio file and write outputs to disk

Requirements (host):  podman, httpx>=0.23  (pip install httpx)
#
# If you want to run on a GPU the host must have appropriate drivers and
# a recent Podman with `--gpus` support (NVIDIA or ROCm).  The --gpus flag is
# passed through by cmd_start; use `--device cuda` in combination with it.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CONTAINER_NAME = "whisperx-server"
IMAGE_NAME = "whisperx-local"
CONTAINER_SOCKET = "/run/api/whisperx.sock"
DEFAULT_SOCKET_DIR = os.environ.get("WHISPERX_SOCKET_DIR", "/tmp/whisperx-api")
DEFAULT_MODELS_DIR = os.path.join(os.getcwd(), "models")

NGINX_CONTAINER_NAME = "whisperx-nginx"


# ── model alias resolution ────────────────────────────────────────────────────


def _build_alias_map(models_dir: str) -> dict[str, str]:
    """Scan models_dir for subdirectories containing an 'alias' file.

    Returns a dict mapping each alias → the container-side absolute path
    of that model directory (i.e. under /models/extra/).

    Example:
        models/kb-whisper-large-ct2/alias  contains "kb-whisper-large"
        → {"kb-whisper-large": "/models/extra/kb-whisper-large-ct2",
           "kb-whisper-large-ct2": "/models/extra/kb-whisper-large-ct2"}

    The directory basename itself is always included as an implicit alias.
    """
    alias_map: dict[str, str] = {}
    root = Path(models_dir)
    if not root.is_dir():
        return alias_map
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        # Skip internal cache/hf directories — only include directories that
        # contain an alias file (i.e. actual model directories).
        alias_file = entry / "alias"
        if not alias_file.is_file():
            continue
        container_path = f"/models/extra/{entry.name}"
        # directory basename is always a valid alias
        alias_map[entry.name] = container_path
        for line in alias_file.read_text().splitlines():
            alias = line.strip()
            if alias and not alias.startswith("#"):
                alias_map[alias] = container_path
    return alias_map


def _resolve_model(value: str | None, models_dir: str, label: str = "model") -> str | None:
    """Resolve a model alias or bare directory name to a container path.

    If *value* already looks like an absolute path or a HuggingFace model ID
    (contains '/' or starts with '/') it is returned as-is.  Otherwise it is
    looked up in the alias map built from *models_dir*.

    Prints a warning and returns *value* unchanged if no alias matches
    (the server will then try to use the value directly, which may fail).
    """
    if value is None:
        return None
    # Absolute path or HF model ID — pass through untouched
    if value.startswith("/") or "/" in value:
        return value
    alias_map = _build_alias_map(models_dir)
    if value in alias_map:
        resolved = alias_map[value]
        print(f"  alias '{value}' → {resolved}")
        return resolved
    print(
        f"  warning: --{label} '{value}' is not an alias or absolute path — "
        "passing to server as-is (may fail if not a valid HF model ID).",
        file=sys.stderr,
    )
    return value


def _print_aliases(models_dir: str) -> None:
    """Print all available model aliases discovered in models_dir."""
    alias_map = _build_alias_map(models_dir)
    if not alias_map:
        print(f"No models found in {models_dir}")
        return
    root = Path(models_dir)
    # Group by container path so each model is shown once
    by_path: dict[str, list[str]] = {}
    for alias, path in sorted(alias_map.items()):
        by_path.setdefault(path, []).append(alias)
    print(f"Available model aliases (from {models_dir}):")
    for path, aliases in sorted(by_path.items()):
        # Parse role/language metadata from the alias file comments
        dirname = path.rsplit("/", 1)[-1]
        alias_file = root / dirname / "alias"
        role = None
        languages: list[str] = []
        if alias_file.is_file():
            for line in alias_file.read_text().splitlines():
                line = line.strip()
                if not line.startswith("#"):
                    continue
                body = line[1:].strip()
                if ":" not in body:
                    continue
                key, _, value = body.partition(":")
                key = key.strip().lower()
                value = value.strip()
                if key == "role" and value:
                    role = value
                elif key in ("language", "languages") and value:
                    languages.append(value)
        meta_parts = [p for p in ([role] + languages) if p]
        header = path + (f"  [{' · '.join(meta_parts)}]" if meta_parts else "")
        print(f"  {header}")
        for a in aliases:
            print(f"    --model {a}")


NGINX_IMAGE_NAME = "whisperx-nginx"
# Resolve once at import time so subcommand functions can use it
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_NGINX_DIR = os.path.join(_PROJECT_ROOT, "container", "nginx")


# ── helpers ───────────────────────────────────────────────────────────────────


def _sock(socket_dir: str) -> str:
    return os.path.join(socket_dir, "whisperx.sock")


def _require_httpx():
    try:
        import httpx

        return httpx
    except ImportError:
        print(
            "error: 'httpx' is not installed on the host.\n" "  Install it with:  pip install 'httpx>=0.23'",
            file=sys.stderr,
        )
        sys.exit(1)


def _client(socket_dir: str, timeout: float = 10.0):
    httpx = _require_httpx()
    return httpx.Client(
        transport=httpx.HTTPTransport(uds=_sock(socket_dir)),
        base_url="http://localhost",
        timeout=timeout,
    )


def _call_health(socket_dir: str) -> dict | None:
    try:
        with _client(socket_dir) as c:
            r = c.get("/health")
            r.raise_for_status()
            return r.json()
    except Exception:
        return None


# ── commands ──────────────────────────────────────────────────────────────────


def cmd_start(args) -> int:
    socket_dir = args.socket_dir
    os.makedirs(socket_dir, exist_ok=True)

    models_dir = os.path.abspath(args.models_dir)
    if not os.path.isdir(models_dir):
        print(f"warning: models directory '{models_dir}' does not exist – container may fail to find models")

    # Resolve model aliases → container-side absolute paths
    if getattr(args, "list_models", False):
        _print_aliases(models_dir)
        return 0
    args.model = _resolve_model(args.model, models_dir, "model")
    args.align_model = _resolve_model(args.align_model, models_dir, "align-model")
    args.diarize_model = _resolve_model(args.diarize_model, models_dir, "diarize-model")

    # Mapping: argparse attribute → container env var
    env_map = {
        "model": "WHISPERX_MODEL",
        "device": "WHISPERX_DEVICE",
        "compute_type": "WHISPERX_COMPUTE_TYPE",
        "language": "WHISPERX_LANGUAGE",
        "batch_size": "WHISPERX_BATCH_SIZE",
        "threads": "WHISPERX_THREADS",
        "beam_size": "WHISPERX_BEAM_SIZE",
        "best_of": "WHISPERX_BEST_OF",
        "vad_method": "WHISPERX_VAD_METHOD",
        "vad_onset": "WHISPERX_VAD_ONSET",
        "vad_offset": "WHISPERX_VAD_OFFSET",
        "chunk_size": "WHISPERX_CHUNK_SIZE",
        "temperature": "WHISPERX_TEMPERATURE",
        "initial_prompt": "WHISPERX_INITIAL_PROMPT",
        "hotwords": "WHISPERX_HOTWORDS",
        "align_model": "WHISPERX_ALIGN_MODEL",
        "diarize_model": "WHISPERX_DIARIZE_MODEL",
        "repetition_penalty": "WHISPERX_REPETITION_PENALTY",
        "no_repeat_ngram_size": "WHISPERX_NO_REPEAT_NGRAM",
        "idle_timeout": "WHISPERX_IDLE_TIMEOUT_SECONDS",
    }
    env_args: list[str] = []
    for attr, env_name in env_map.items():
        val = getattr(args, attr, None)
        if val is not None:
            env_args += ["-e", f"{env_name}={val}"]

    hf_token = getattr(args, "hf_token", None) or os.environ.get("HF_TOKEN")
    if hf_token:
        env_args += ["-e", f"HF_TOKEN={hf_token}"]

    podman_cmd = [
        "podman",
        "run",
        "--detach",
        "--name",
        CONTAINER_NAME,
        "--replace",  # remove any container with the same name first
        "--network=none",
        # Drop all Linux capabilities — whisperx only needs plain Python/PyTorch.
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
    ]
    # GPU passthrough (podman 4.6+ supports --gpus like Docker).
    # In practice some images (including our CUDA base) expect the device nodes
    # to be visible via the old `--device nvidia.com/gpu=…` syntax.  Add that
    # automatically alongside --gpus so the command works consistently.
    if getattr(args, "gpus", None):
        # the value may be "all" or a comma-separated list
        podman_cmd += ["--gpus", str(args.gpus)]
        podman_cmd += ["--device", f"nvidia.com/gpu={args.gpus}"]
    podman_cmd += [
        # UDS socket directory (server creates the .sock file inside here)
        "-v",
        f"{socket_dir}:{os.path.dirname(CONTAINER_SOCKET)}:z",
        # all models live here as plain directories – read-only
        "-v",
        f"{models_dir}:/models/extra:ro",
        "-e",
        "TORCH_HOME=/models/extra/cache/torch",
    ]
    if getattr(args, "dev", False):
        # Mount the host-side Python scripts directly into the container so
        # edits take effect on the next restart without a full image rebuild.
        container_dir = os.path.join(_PROJECT_ROOT, "container")
        for script in ("server.py", "client.py"):
            host_path = os.path.join(container_dir, script)
            if os.path.isfile(host_path):
                podman_cmd += ["-v", f"{host_path}:/app/container/{script}:ro,z"]
        print("  [dev] mounting host container/*.py over image copies")
    podman_cmd = podman_cmd + env_args + [IMAGE_NAME]

    print(f"Starting container '{CONTAINER_NAME}' from image '{IMAGE_NAME}' …")
    result = subprocess.run(podman_cmd)
    if result.returncode != 0:
        print("Failed to start container.", file=sys.stderr)
        return result.returncode

    sock = _sock(socket_dir)
    print(f"Waiting for server to become ready at {sock}", end="", flush=True)
    for _ in range(180):  # up to 3 minutes (model load can be slow on CPU)
        time.sleep(1)
        # check whether the container is still running; if it died we should bail
        status = subprocess.run(
            ["podman", "inspect", "--format", "{{.State.Status}}", CONTAINER_NAME],
            capture_output=True,
            text=True,
        )
        if status.returncode == 0:
            state = status.stdout.strip()
            if state not in ("running", "created"):
                # container exited or errored before socket appeared
                print("\ncontainer exited unexpectedly (state={})".format(state), file=sys.stderr)
                print(f"Check logs with:  podman logs {CONTAINER_NAME}", file=sys.stderr)
                return 1
        print(".", end="", flush=True)
        if Path(sock).exists():
            health = _call_health(socket_dir)
            # Server is ready if:
            # 1. A model is loaded (ready=true), OR
            # 2. Socket exists and server is responding (even without a model, which is valid)
            if health:
                if health.get("ready") or not health.get("model"):
                    print(" ready!")
                    print(
                        f"  model   : {health.get('model') or 'none (use POST /reload to load)'}\n"
                        f"  device  : {health.get('device')} ({health.get('compute_type')})\n"
                        f"  language: {health.get('language') or 'auto-detect'}"
                    )
                    return 0
    print(
        "\ntimeout – server did not become ready.\n" f"Check logs with:  podman logs {CONTAINER_NAME}",
        file=sys.stderr,
    )
    return 1


def cmd_stop(args) -> int:
    result = subprocess.run(
        ["podman", "stop", CONTAINER_NAME],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"Stopped '{CONTAINER_NAME}'.")
    else:
        print(result.stderr.strip() or f"Could not stop '{CONTAINER_NAME}'", file=sys.stderr)
    return result.returncode


def cmd_status(args) -> int:
    sock = _sock(args.socket_dir)
    if not Path(sock).exists():
        print(f"Socket not found at {sock} – server does not appear to be running.")
        return 1
    health = _call_health(args.socket_dir)
    if health:
        print(json.dumps(health, indent=2))
        return 0
    print("Socket exists but server is not responding.", file=sys.stderr)
    return 1


def cmd_reload(args) -> int:
    """
    Ask the running server to reload the ASR (and VAD) model with new
    settings without restarting the container.

    Only pass the settings you want to change; omitted fields keep their
    current values.  Alignment and diarization caches are also cleared.
    """
    models_dir = os.path.abspath(os.path.join(_PROJECT_ROOT, "models"))
    if getattr(args, "list_models", False):
        _print_aliases(models_dir)
        return 0
    args.model = _resolve_model(args.model, models_dir, "model")
    args.align_model = _resolve_model(getattr(args, "align_model", None), models_dir, "align-model")
    args.diarize_model = _resolve_model(getattr(args, "diarize_model", None), models_dir, "diarize-model")

    payload: dict = {}
    for attr in (
        "model",
        "device",
        "compute_type",
        "language",
        "batch_size",
        "threads",
        "beam_size",
        "best_of",
        "vad_method",
        "vad_onset",
        "vad_offset",
        "chunk_size",
        "temperature",
        "repetition_penalty",
        "no_repeat_ngram_size",
        "initial_prompt",
        "hotwords",
        "align_model",
        "diarize_model",
    ):
        val = getattr(args, attr, None)
        if val is not None:
            payload[attr] = val

    httpx = _require_httpx()
    try:
        with _client(args.socket_dir, timeout=300.0) as c:
            print("Requesting model reload … (this may take a while)")
            r = c.post("/reload", json=payload)
            r.raise_for_status()
            print(json.dumps(r.json(), indent=2))
        return 0
    except httpx.HTTPStatusError as exc:
        print(f"Server error {exc.response.status_code}: {exc.response.text}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def cmd_transcribe(args) -> int:
    """
    Send an audio file to the running server and write the formatted
    outputs to --output-dir on the host.  The container itself never
    writes to disk.
    """
    audio_path = os.path.abspath(args.audio)
    if not os.path.isfile(audio_path):
        print(f"error: audio file '{audio_path}' not found", file=sys.stderr)
        return 1

    # Build per-request params (only include what was explicitly set)
    params: dict = {}
    opt_map = {
        "language": "language",
        "task": "task",
        "batch_size": "batch_size",
        "chunk_size": "chunk_size",
        "diarize": "diarize",
        "min_speakers": "min_speakers",
        "max_speakers": "max_speakers",
        "diarize_model": "diarize_model",
        "no_align": "no_align",
        "align_model": "align_model",
        "output_format": "output_format",
        "interpolate_method": "interpolate_method",
        "return_char_alignments": "return_char_alignments",
        "speaker_embeddings": "speaker_embeddings",
        "verbose": "verbose",
        "print_progress": "print_progress",
        "highlight_words": "highlight_words",
        "max_line_width": "max_line_width",
        "max_line_count": "max_line_count",
    }
    for attr, param_key in opt_map.items():
        val = getattr(args, attr, None)
        # include booleans only when explicitly True; skip None/False
        if val is not None and val is not False:
            params[param_key] = val

    output_dir = Path(args.output_dir or "output")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_basename = Path(audio_path).stem

    httpx = _require_httpx()
    try:
        with _client(args.socket_dir, timeout=3600.0) as c:
            print(f"Sending '{audio_path}' to server …")
            with open(audio_path, "rb") as f:
                r = c.post(
                    "/transcribe",
                    files={"audio": (os.path.basename(audio_path), f)},
                    data={"params": json.dumps(params)},
                )
            r.raise_for_status()
            data = r.json()

        print(
            f"Done in {data.get('duration_seconds', '?')}s  "
            f"| language: {data.get('language', '?')}  "
            f"| segments: {len(data.get('segments', []))}"
        )
        for fmt, content in data.get("outputs", {}).items():
            out_file = output_dir / f"{audio_basename}.{fmt}"
            out_file.write_text(content, encoding="utf-8")
            print(f"  wrote {out_file}")
        return 0

    except httpx.HTTPStatusError as exc:
        print(f"Server error {exc.response.status_code}: {exc.response.text}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


# ── argument parser ───────────────────────────────────────────────────────────


def _add_model_config_args(p: argparse.ArgumentParser) -> None:
    """Add model-config arguments shared by 'start' and 'reload'."""
    p.add_argument("--model", default=None, help="Whisper model name or HF path")
    p.add_argument("--device", default=None, choices=["cpu", "cuda"])
    p.add_argument(
        "--compute-type", dest="compute_type", default=None, choices=["default", "float16", "float32", "int8"]
    )
    p.add_argument("--language", default=None, help="ISO-639-1 code (omit for auto-detect)")
    p.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--beam-size", dest="beam_size", type=int, default=None)
    p.add_argument("--best-of", dest="best_of", type=int, default=None)
    p.add_argument("--vad-method", dest="vad_method", default=None, choices=["pyannote", "silero"])
    p.add_argument("--vad-onset", dest="vad_onset", type=float, default=None)
    p.add_argument("--vad-offset", dest="vad_offset", type=float, default=None)
    p.add_argument("--chunk-size", dest="chunk_size", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--initial-prompt", dest="initial_prompt", default=None)
    p.add_argument("--hotwords", default=None)
    p.add_argument(
        "--repetition-penalty",
        dest="repetition_penalty",
        type=float,
        default=None,
        help="Decoding repetition penalty (faster-whisper TranscriptionOptions.repetition_penalty).",
    )
    p.add_argument(
        "--idle-timeout",
        dest="idle_timeout",
        type=int,
        default=None,
        help="Seconds of idle time after which models are unloaded (0 disables).",
    )
    p.add_argument(
        "--no-repeat-ngram-size",
        dest="no_repeat_ngram_size",
        type=int,
        default=None,
        help=(
            "Prevent n-gram repetition of this size during decoding. "
            "0 = disabled. Try 2 or 3 to reduce hallucinations."
        ),
    )
    p.add_argument(
        "--align-model",
        dest="align_model",
        default=None,
        help="Alignment model name or local path (default: auto per language)",
    )
    p.add_argument(
        "--diarize-model", dest="diarize_model", default=None, help="Diarization pipeline name or local path"
    )


# ── nginx sidecar commands ───────────────────────────────────────────────────


def cmd_build_nginx(_args) -> int:
    """Build the whisperx-nginx image from container/nginx/."""
    if not os.path.isdir(_NGINX_DIR):
        print(f"nginx directory not found: {_NGINX_DIR}", file=sys.stderr)
        return 1
    containerfile = os.path.join(_NGINX_DIR, "Containerfile")
    cmd = [
        "podman",
        "build",
        "-t",
        NGINX_IMAGE_NAME,
        "-f",
        containerfile,
        _NGINX_DIR,
    ]
    print(f"Building nginx image '{NGINX_IMAGE_NAME}' from {_NGINX_DIR} …")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print(f"Image '{NGINX_IMAGE_NAME}' built successfully.")
    return result.returncode


def cmd_start_nginx(args) -> int:
    """Start the nginx sidecar container.

    The container:
      • runs entirely as the unprivileged 'nginx' user (uid 101)
      • has --cap-drop=ALL  (no Linux capabilities whatsoever)
      • has a read-only root filesystem (only /tmp is writable via tmpfs)
      • is NOT started with --network=none so it can accept TCP connections
      • proxies every request to the whisperx UDS socket
    """
    socket_dir = args.socket_dir
    if not os.path.isdir(socket_dir):
        print(
            f"Socket directory '{socket_dir}' does not exist.\n"
            "Start the whisperx server first:  python container/manage.py start",
            file=sys.stderr,
        )
        return 1

    listen_host = args.listen_host
    port = args.port

    cmd = [
        "podman",
        "run",
        "--detach",
        "--name",
        NGINX_CONTAINER_NAME,
        "--replace",
        # ── minimal rights ──────────────────────────────────────────────
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        "--read-only",
        # nginx writes pid / temp files to /tmp; everything else is read-only
        "--tmpfs",
        "/tmp:mode=1777",
        # ── socket access ───────────────────────────────────────────────
        # Mount the socket directory so nginx can connect() to the socket.
        # Needs to be rw because connect() on a Unix socket requires write
        # permission on the socket file itself.
        "-v",
        f"{socket_dir}:/run/api:z",
        # ── network ─────────────────────────────────────────────────────
        # nginx always listens on 8080 inside the container (hard-coded in
        # nginx.conf).  The --port flag controls only which host port is
        # exposed; map it to the fixed container port 8080.
        "-p",
        f"{listen_host}:{port}:8080",
        NGINX_IMAGE_NAME,
    ]

    print(
        f"Starting nginx sidecar '{NGINX_CONTAINER_NAME}' …\n"
        f"  Listening on  {listen_host}:{port}\n"
        f"  Socket dir    {socket_dir}\n"
        f"  Image         {NGINX_IMAGE_NAME}"
    )
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Failed to start nginx container.", file=sys.stderr)
        return result.returncode

    # Quick sanity-check: wait a couple of seconds then hit the health endpoint
    # through the TCP port (not the UDS) to confirm the full proxy path works.
    for attempt in range(10):
        time.sleep(1)
        try:
            import httpx

            r = httpx.get(f"http://{listen_host if listen_host != '0.0.0.0' else '127.0.0.1'}:{port}/health", timeout=3)
            if r.status_code == 200:
                data = r.json()
                print(f"nginx proxy is up. Whisperx backend: " f"ready={data.get('ready')}, model={data.get('model')}")
                print(
                    f"\nTest from this machine:\n"
                    f"  curl http://localhost:{port}/health\n"
                    f"Test from another machine (if listen_host is 0.0.0.0):\n"
                    f"  curl http://<this-host-ip>:{port}/health"
                )
                return 0
        except Exception:
            pass
    print("nginx started but health check did not respond — check logs:", file=sys.stderr)
    print(f"  podman logs {NGINX_CONTAINER_NAME}", file=sys.stderr)
    return 1


def cmd_stop_nginx(_args) -> int:
    """Stop and remove the nginx sidecar container."""
    result = subprocess.run(
        ["podman", "stop", NGINX_CONTAINER_NAME],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"Stopped '{NGINX_CONTAINER_NAME}'.")
    else:
        print(result.stderr.strip() or f"Could not stop '{NGINX_CONTAINER_NAME}'", file=sys.stderr)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage the whisperx-server container",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--socket-dir",
        dest="socket_dir",
        default=DEFAULT_SOCKET_DIR,
        help="Host directory shared with container for the Unix socket",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── start ─────────────────────────────────────────────────────────────
    p_start = sub.add_parser("start", help="Start the whisperx server container")
    p_start.add_argument(
        "--models-dir",
        dest="models_dir",
        default=DEFAULT_MODELS_DIR,
        help="Host path to the models directory (torch cache, alignment models; mounted read-only)",
    )
    p_start.add_argument(
        "--dev",
        action="store_true",
        default=False,
        help="Mount host container/server.py and container/client.py into the container at startup "
        "so code changes take effect on restart without rebuilding the image.",
    )
    p_start.add_argument(
        "--hf-token", dest="hf_token", default=None, help="HuggingFace token (falls back to $HF_TOKEN)"
    )
    p_start.add_argument(
        "--list-models",
        dest="list_models",
        action="store_true",
        default=False,
        help="List available model aliases from the models directory and exit",
    )
    p_start.add_argument(
        "--gpus",
        dest="gpus",
        default=None,
        help=(
            "GPU devices to expose to the container. ``all`` passes all visible GPUs,"
            " or use a comma-separated list like ``0,1``.  Requires a recent Podman with"
            " NVIDIA/AMD support; omit to run on CPU only.  The script also adds a"
            " matching ``--device nvidia.com/gpu=…`` binding for compatibility."
        ),
    )
    _add_model_config_args(p_start)

    # ── stop ──────────────────────────────────────────────────────────────
    sub.add_parser("stop", help="Stop the whisperx server container")

    # ── status ────────────────────────────────────────────────────────────
    sub.add_parser("status", help="Show server health and current model info")

    # ── reload ────────────────────────────────────────────────────────────
    p_reload = sub.add_parser(
        "reload",
        help="Reload the ASR model with new settings (no container restart)",
        description=(
            "Reload the ASR/VAD model inside the running container.\n"
            "Only pass the fields you want to change; others keep their current value.\n"
            "Per-request params (language, diarize, output_format …) never need reload."
        ),
    )
    _add_model_config_args(p_reload)

    # ── transcribe ────────────────────────────────────────────────────────
    p_tx = sub.add_parser("transcribe", help="Transcribe an audio file via the running server")
    p_tx.add_argument("audio", help="Path to the audio file to transcribe")
    p_tx.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Directory to write output files (default: ./output)",
    )
    p_tx.add_argument(
        "-f",
        "--output-format",
        dest="output_format",
        default="all",
        choices=["all", "srt", "vtt", "txt", "tsv", "json", "aud"],
    )
    p_tx.add_argument("--language", default=None)
    p_tx.add_argument("--task", default=None, choices=["transcribe", "translate"])
    p_tx.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    p_tx.add_argument("--chunk-size", dest="chunk_size", type=int, default=None)
    p_tx.add_argument("--diarize", action="store_true", default=None)
    p_tx.add_argument("--min-speakers", dest="min_speakers", type=int, default=None)
    p_tx.add_argument("--max-speakers", dest="max_speakers", type=int, default=None)
    p_tx.add_argument("--diarize-model", dest="diarize_model", default=None)
    p_tx.add_argument("--no-align", dest="no_align", action="store_true", default=None)
    p_tx.add_argument("--align-model", dest="align_model", default=None)
    p_tx.add_argument(
        "--interpolate-method", dest="interpolate_method", default=None, choices=["nearest", "linear", "ignore"]
    )
    p_tx.add_argument("--return-char-alignments", dest="return_char_alignments", action="store_true", default=None)
    p_tx.add_argument("--speaker-embeddings", dest="speaker_embeddings", action="store_true", default=None)
    p_tx.add_argument("--verbose", action="store_true", default=None)
    p_tx.add_argument("--print-progress", dest="print_progress", action="store_true", default=None)
    p_tx.add_argument("--highlight-words", dest="highlight_words", action="store_true", default=None)
    p_tx.add_argument("--max-line-width", dest="max_line_width", type=int, default=None)
    p_tx.add_argument("--max-line-count", dest="max_line_count", type=int, default=None)

    # ── build-nginx ───────────────────────────────────────────────────────
    sub.add_parser(
        "build-nginx",
        help=f"Build the '{NGINX_IMAGE_NAME}' image from container/nginx/",
    )

    # ── start-nginx ───────────────────────────────────────────────────────
    p_nginx = sub.add_parser(
        "start-nginx",
        help="Start the nginx reverse-proxy sidecar in front of the whisperx server",
        description=(
            "Runs a minimal nginx container that proxies TCP connections to the\n"
            "whisperx Unix Domain Socket.  The nginx container runs fully as\n"
            "an unprivileged user with --cap-drop=ALL and a read-only root fs.\n"
            "\n"
            "The whisperx server must already be running (manage.py start).\n"
            "\n"
            "To expose the API on all network interfaces (so other machines can\n"
            "reach it), pass --listen-host 0.0.0.0.  Default is 127.0.0.1 which\n"
            "only accepts connections from localhost."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_nginx.add_argument(
        "--listen-host",
        dest="listen_host",
        default="127.0.0.1",
        metavar="HOST",
        help=(
            "IP address to listen on.  Use 0.0.0.0 to accept connections from "
            "any interface / other machines on the network.  "
            "Default: 127.0.0.1 (localhost only)."
        ),
    )
    p_nginx.add_argument(
        "--port",
        dest="port",
        type=int,
        default=8080,
        help="TCP port to listen on inside the nginx container (default: 8080)",
    )

    # ── stop-nginx ────────────────────────────────────────────────────────
    sub.add_parser("stop-nginx", help=f"Stop the '{NGINX_CONTAINER_NAME}' container")

    args = parser.parse_args()
    dispatch = {
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "reload": cmd_reload,
        "transcribe": cmd_transcribe,
        "build-nginx": cmd_build_nginx,
        "start-nginx": cmd_start_nginx,
        "stop-nginx": cmd_stop_nginx,
    }
    sys.exit(dispatch[args.command](args))


if __name__ == "__main__":
    main()
