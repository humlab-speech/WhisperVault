#!/usr/bin/env python3
"""Entrypoint helper for the whisperx container.

This script forwards arguments to the whisperx CLI and ensures that model
caches are mapped to mounted volumes. By default it operates in offline
mode unless the environment variable OFFLINE is set to '0'.

Usage (host side):

    podman run --rm \
        -v /path/to/hf-cache:/models/hf:z \
        -v /path/to/input:/input:z \
        -v /path/to/output:/output:z \
        whisperx-local \
        /input/audio.wav --model large-v2 --diarize --output_dir /output

The script merely sets the appropriate HF_* environment variables and
invokes the real CLI.
"""

import os
import sys

# configure cache locations inside container
os.environ.setdefault("HF_HOME", "/models/hf")
os.environ.setdefault("HF_HUB_CACHE", "/models/hf")
os.environ.setdefault("XDG_CACHE_HOME", "/models/cache")
# prevent matplotlib from attempting to write into the read-only /models volume
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
# also direct torch/torchaudio caches to shared volume
os.environ.setdefault("TORCH_HOME", "/models/cache/torch")

# enforce offline mode by default
if os.environ.get("OFFLINE", "1") != "0":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

# delegate to whisperx CLI
from whisperx.__main__ import cli

# when cli() is called without arguments, it reads sys.argv itself
sys.exit(cli())
