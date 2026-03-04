#!/usr/bin/env python3
"""
transcribe.py — minimal CLI client for the WhisperX API server.

Sends an audio file to the running whisperx container and writes the
output to disk (or prints it).  Communicates over the Unix Domain Socket
so the container can stay --network=none.

Requirements
------------
    pip install httpx

Usage examples
--------------
    # Basic transcription (txt output, language auto-detected):
    python transcribe.py audio.wav

    # Swedish, SRT output, written to ./output/:
    python transcribe.py audio.wav --language sv --format srt --output-dir output

    # With speaker diarization, all formats:
    python transcribe.py audio.wav --language sv --diarize

    # Multiple formats at once:
    python transcribe.py audio.wav --language sv --format srt txt

    # Just print the plain text to stdout, don't write files:
    python transcribe.py audio.wav --language sv --format txt --print

    # Use a different socket path:
    python transcribe.py audio.wav --socket /tmp/whisperx-api/whisperx.sock

    # Check which model is loaded and what models are available:
    python transcribe.py --status
    python transcribe.py --models
"""

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("httpx is required:  pip install httpx")

DEFAULT_SOCKET = os.environ.get("WHISPERX_SOCKET", "/tmp/whisperx-api/whisperx.sock")


def make_client(socket_path: str) -> httpx.Client:
    return httpx.Client(
        transport=httpx.HTTPTransport(uds=socket_path),
        base_url="http://localhost",
        timeout=None,  # transcription can take many minutes on CPU
    )


def cmd_status(client: httpx.Client) -> None:
    r = client.get("/health")
    r.raise_for_status()
    h = r.json()
    print(f"ready       : {h['ready']}")
    print(f"model       : {h['model']}")
    print(f"device      : {h['device']}  ({h['compute_type']})")
    print(f"language    : {h['language'] or 'auto-detect'}")
    print(f"vad_method  : {h['vad_method']}")
    if h["align_models_cached"]:
        print(f"align cached: {', '.join(h['align_models_cached'])}")
    if h["diarize_pipelines_cached"]:
        print(f"diarize cached: {', '.join(h['diarize_pipelines_cached'])}")


def cmd_models(client: httpx.Client) -> None:
    r = client.get("/models")
    r.raise_for_status()
    data = r.json()
    loaded_asr = data["currently_loaded"]["asr"]
    print(f"{'ROLE':<14} {'MODEL ID':<55} STATUS")
    print("-" * 80)
    for m in data["available"]:
        status = "● loaded" if m["model_id"] == loaded_asr else ""
        print(f"{m['role']:<14} {m['model_id']:<55} {status}")
    print()
    align_loaded = data["currently_loaded"]["alignment"]
    diarize_loaded = data["currently_loaded"]["diarization"]
    if align_loaded:
        print(f"Alignment pipelines in memory : {', '.join(align_loaded)}")
    if diarize_loaded:
        print(f"Diarization pipelines in memory: {', '.join(diarize_loaded)}")


def cmd_transcribe(client: httpx.Client, args: argparse.Namespace) -> None:
    audio_path = Path(args.audio)
    if not audio_path.exists():
        sys.exit(f"File not found: {audio_path}")

    # Build the params JSON sent alongside the audio file
    params: dict = {}
    if args.language:
        params["language"] = args.language
    if args.task:
        params["task"] = args.task
    if args.diarize:
        params["diarize"] = True
    if args.min_speakers is not None:
        params["min_speakers"] = args.min_speakers
    if args.max_speakers is not None:
        params["max_speakers"] = args.max_speakers
    if args.no_align:
        params["no_align"] = True
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size

    # Resolve output formats
    formats = args.format if args.format else ["txt"]
    params["output_format"] = formats if len(formats) > 1 else formats[0]

    if args.highlight_words:
        params["highlight_words"] = True
    if args.max_line_width is not None:
        params["max_line_width"] = args.max_line_width
    if args.max_line_count is not None:
        params["max_line_count"] = args.max_line_count

    print(f"Transcribing: {audio_path.name}", file=sys.stderr)
    print(f"Params      : {json.dumps(params)}", file=sys.stderr)

    with audio_path.open("rb") as fh:
        r = client.post(
            "/transcribe",
            files={"audio": (audio_path.name, fh, "application/octet-stream")},
            data={"params": json.dumps(params)},
        )

    if r.status_code != 200:
        sys.exit(f"Server error {r.status_code}: {r.text}")

    result = r.json()
    print(
        f"Done in {result['duration_seconds']}s  |  "
        f"language: {result['language']}  |  "
        f"segments: {len(result['segments'])}",
        file=sys.stderr,
    )

    speakers = sorted({s.get("speaker") for s in result["segments"] if s.get("speaker")})
    if speakers:
        print(f"Speakers    : {', '.join(speakers)}", file=sys.stderr)

    outputs: dict[str, str] = result.get("outputs", {})

    # Determine where to write files
    output_dir = Path(args.output_dir) if args.output_dir else None
    stem = audio_path.stem

    for fmt, content in outputs.items():
        if args.print:
            # --print: write to stdout, skip file output
            if len(outputs) > 1:
                print(f"\n─── {fmt.upper()} ───")
            print(content)
        else:
            dest_dir = output_dir or audio_path.parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            out_file = dest_dir / f"{stem}.{fmt}"
            out_file.write_text(content, encoding="utf-8")
            print(f"Written: {out_file}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe audio via the WhisperX API server (Unix socket).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--socket",
        default=DEFAULT_SOCKET,
        metavar="PATH",
        help="Unix socket path (default: $WHISPERX_SOCKET or /tmp/whisperx-api/whisperx.sock)",
    )

    # ── info subcommands (no audio file needed) ──────────────────────────────
    info = parser.add_mutually_exclusive_group()
    info.add_argument(
        "--status",
        action="store_true",
        help="Print server health and currently loaded model, then exit",
    )
    info.add_argument(
        "--models",
        action="store_true",
        help="List all models available in the server's cache, then exit",
    )

    # ── audio ────────────────────────────────────────────────────────────────
    parser.add_argument(
        "audio",
        nargs="?",
        help="Path to the audio file to transcribe",
    )

    # ── transcription options ────────────────────────────────────────────────
    parser.add_argument(
        "--language",
        "-l",
        default=None,
        metavar="CODE",
        help="ISO-639-1 language code (e.g. sv, en).  Omit for auto-detect.",
    )
    parser.add_argument(
        "--task", default=None, choices=["transcribe", "translate"], help="'translate' forces English output"
    )
    parser.add_argument("--diarize", "-d", action="store_true", help="Enable speaker diarization")
    parser.add_argument("--min-speakers", dest="min_speakers", type=int, default=None)
    parser.add_argument("--max-speakers", dest="max_speakers", type=int, default=None)
    parser.add_argument(
        "--no-align", dest="no_align", action="store_true", help="Skip forced word-level alignment (faster)"
    )
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)

    # ── output options ───────────────────────────────────────────────────────
    parser.add_argument(
        "--format",
        "-f",
        nargs="+",
        default=None,
        metavar="FMT",
        choices=["txt", "srt", "vtt", "tsv", "json", "aud"],
        help="Output format(s): txt srt vtt tsv json aud  (default: txt)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        dest="output_dir",
        default=None,
        metavar="DIR",
        help="Directory to write output files (default: same directory as audio file)",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print output to stdout instead of writing files",
    )
    parser.add_argument("--highlight-words", dest="highlight_words", action="store_true")
    parser.add_argument("--max-line-width", dest="max_line_width", type=int, default=None)
    parser.add_argument("--max-line-count", dest="max_line_count", type=int, default=None)

    args = parser.parse_args()

    if not os.path.exists(args.socket):
        sys.exit(
            f"Socket not found: {args.socket}\n"
            "Is the server running?  Start it with:\n"
            "  python container/manage.py start --model KBLab/kb-whisper-large "
            "--device cpu --compute-type float32 --language sv"
        )

    with make_client(args.socket) as client:
        if args.status:
            cmd_status(client)
        elif args.models:
            cmd_models(client)
        elif args.audio:
            cmd_transcribe(client, args)
        else:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
