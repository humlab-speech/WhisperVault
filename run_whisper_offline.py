#!/usr/bin/python3
# run with system python (avoids pyenv shims); if a venv python is
# available we immediately re-exec with it so everything runs in the
# project's virtualenv.
import os, sys
venv = os.path.join(os.getcwd(), ".venv", "bin", "python3")
if os.path.isfile(venv) and os.path.abspath(sys.executable) != os.path.abspath(venv):
    os.execv(venv, [venv] + sys.argv)

"""Convenience wrapper for running the whisperx-local podman image offline.

This script takes a single audio path followed by any arguments that
should be passed to the embedded :command:`whisperx` CLI.  It
automatically adds sane defaults (e.g. ``--best_of 10``) and arranges
the podman command with ``--network=none`` plus the minimal mounts
required for cache and I/O.

Usage::

    python container/run_whisper_offline.py \
        /path/to/audio.wav \
        --model KBLab/kb-whisper-large --device cpu --diarize \
        --output_dir /tmp/out --output_format srt

Any familiar whisperx options are supported; unrecognised flags are
forwarded verbatim.  If you omit ``--best_of`` the script will inject a
value of 10.  ``HF_TOKEN`` may be exported in your environment if a
protected diarization model is needed.

Note: the host must have a HuggingFace cache directory (e.g.
``~/.cache/huggingface/hub``) and, optionally, a torch cache with any
alignment models you intend to use.  Those directories will be mounted
read‑only into the container.  See ``container/README.md`` for details.
"""

import argparse
import os
import shlex
import subprocess
import sys


def build_podman_command(audio_file: str, extra_args: list[str]) -> list[str]:
    # make sure audio_file is absolute as well; callers should already
    # have normalized it but double‑check anyway.
    audio_file = os.path.abspath(audio_file)

    # default output directory (can be overridden on command line)
    output_dir = os.environ.get("OUTPUT_DIR", os.path.join(os.getcwd(), "output"))
    os.makedirs(output_dir, exist_ok=True)

    # ensure best_of default
    if not any(arg.startswith("--best_of") for arg in extra_args):
        extra_args = ["--best_of", "10"] + extra_args

    # any output_dir given by the caller must be rewritten to the
    # container's mount point (/output).  we'll strip both the flag and
    # its value, then append ours.
    cleaned_args: list[str] = []
    skip_next = False
    for arg in extra_args:
        if skip_next:
            skip_next = False
            continue
        if arg.startswith("--output_dir") or arg.startswith("--output-dir"):
            # if form --output_dir=foo handle separately
            if "=" in arg:
                # drop entirely
                continue
            else:
                # skip the next item as well (the value)
                skip_next = True
                continue
        cleaned_args.append(arg)
    extra_args = cleaned_args + ["--output_dir", "/output"]

    cmd = [
        "podman",
        "run",
        "--rm",
        "--network=none",
        "-v",
        f"{os.getcwd()}/models:/models:ro",
        "-v",
        f"{os.path.expanduser('~')}/.cache/huggingface/hub:/models/hf:ro",
        "-v",
        f"{audio_file}:/input/{os.path.basename(audio_file)}:ro",
        "-v",
        f"{output_dir}:/output",
        "whisperx-local",
        "standalone",               # tell server.py to use CLI passthrough mode
        f"/input/{os.path.basename(audio_file)}",
    ]
    cmd.extend(extra_args)
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the whisperx-local container with network=none",
        add_help=False,
    )
    parser.add_argument("audio", help="path to audio file to transcribe")
    # parse known arguments then leave the rest for forwarding
    args, remaining = parser.parse_known_args()

    # convert the audio path to an absolute filename so that podman
    # mounts it correctly; relative paths end up creating a directory
    # inside the container which confuses ffmpeg.
    args.audio = os.path.abspath(args.audio)

    if not os.path.isfile(args.audio):
        print(f"error: audio file '{args.audio}' not found", file=sys.stderr)
        sys.exit(1)

    podman_cmd = build_podman_command(args.audio, remaining)
    # print the command for debugging if desired
    print("running:", shlex.join(podman_cmd))

    # execute
    result = subprocess.run(podman_cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    # warn if running under a pyenv shim, which can fail when pyenv isn't
    # installed on a clean system.  The wrapper works with any Python 3
    # interpreter, so you may invoke it explicitly via your venv if needed.
    if "pyenv" in sys.executable:
        print(
            f"[warning] running under pyenv shim ({sys.executable}). "
            "Please use a real Python 3 interpreter (e.g. './.venv/bin/python') "
            "or install pyenv if you really need it.",
            file=sys.stderr,
        )
    main()
