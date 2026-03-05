# WhisperX Container

This subfolder provides a lightweight container image that bundles the
WhisperX transcription/diarization code and makes it easy to run inside
`podman` or `docker` with model cache and output directories mounted from
the host.

## Building the image

From the repository root:

```bash
podman build -t whisperx-local -f container/Containerfile .
```

The resulting image contains all Python dependencies and the full project
source.  It does **not** include any model weights – these must be mounted
at runtime from the host cache.

## Directory layout inside container

- `/app` – cloned repository, entrypoint script, and Python install
- `/models/hf` – hf hub cache (ideally the project-local copy under `./models/hf`; the container no longer mounts your global `~/.cache` by default)
- `/models/cache` – general cache location (the container writes nothing here but it is used by
  `TORCH_HOME` to store the alignment model and other torch assets when mounted read-only)
- `/input` – audio files you wish to transcribe
- `/output` – transcription results will be written here

## Entry point: `manage.py`

`manage.py` simply sets a few environment variables and then invokes the
standard `whisperx` CLI.  It forces offline mode by default (`HF_HUB_OFFLINE=1`)
so you can run the container with `--network=none` and guarantee zero
network traffic once the models are cached.

Any arguments you supply after the audio file are passed through unchanged:

> **Note:** the image already contains the NLTK `punkt`/`punkt_tab` data required for
> alignment so you don't need network access during transcription.  Offline mode is
> enforced by default, but if you plan to use gated diarization models you can provide
> an HF token via `-e HF_TOKEN=…` on the `podman run` command line.

```bash
podman run --rm \
    -v $PWD/models:/models:ro \
    -v /path/to/input:/input:z \
    -v /path/to/output:/output:z \
    whisperx-local \
    /input/ZOOM0020_LR.wav \
    --model KBLab/kb-whisper-large \
    --device cpu --compute_type float32 \
    --diarize --output_dir /output --output_format txt
```

The mounted `models` volume allows reuse of downloaded models between
runs; the container itself never writes to its own image.  This also means
you can populate `/models/hf` on the host prior to running, allowing
completely offline operation.

You should also pre‑cache any torch/torchaudio assets used during
alignment (e.g. the wav2vec2 model).  For example:

```bash
export TORCH_HOME=$PWD/models/cache/torch
python - <<'PYTHON'
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H
WAV2VEC2_ASR_BASE_960H.get_model()
PYTHON
```

The downloaded `.pth` file will live under `models/cache/torch/hub/checkpoints`
and will be visible inside the container when you mount the `models` directory.

> **Warning**: you may see a startup message about `torchcodec` not being
> installed correctly.  This is harmless (audio decoding will still work via
> PyTorch/ffmpeg) and is due to a library version mismatch.  You can ignore it
> or remove the `torchcodec` package in your own image if desired.

## Command‑line arguments

The container does not offer its own argument set; it simply exposes the
underlying `whisperx` CLI.  Useful flags include:

- `--model` – specify the Whisper model (e.g. `small`, `large-v2`, `KBLab/kb-whisper-large`).
- `--device` / `--compute_type` – usually `cpu` + `float32` for containers.
- `--diarize` – enable speaker diarization (uses `pyannote/speaker-diarization-community-1`).
- `--output_dir`, `--output_format` – control where results are written.
- `--language` – override language detection if desired.
- `--hf_token` – supply HF token for gated pyannote models (or export in
  environment before starting container).

Any other `whisperx` options may be used as well.

## Offline / networkless execution

To run with zero network access:

```bash
podman run --rm --network=none \
    -v $PWD/models:/models:ro \
    -v /path/to/input:/input:z \
    -v /path/to/output:/output:z \
    whisperx-local \
    /input/ZOOM0020_LR.wav \
    --model KBLab/kb-whisper-large \
    --device cpu --compute_type float32 \
    --diarize --output_dir /output --output_format txt
```

because the entrypoint sets `HF_HUB_OFFLINE=1` by default, the container
will refuse any network access even if the runtime allowed it.  We also set
`MPLCONFIGDIR=/tmp/matplotlib` inside the image so matplotlib won't attempt to
write into the read‑only volume.  If you do need networking, set `OFFLINE=0`
in the podman environment.

## Example run (local test)

Below we demonstrate transcribing the existing `ZOOM0020_LR.wav` file with
network disabled.  First ensure the host cache contains the previously
downloaded models (the repo has them under `~/.cache/huggingface/hub`).

```bash
podman build -t whisperx-local -f container/Containerfile .

# long-form invocation (what the script generates for you):
podman run --rm --network=none \
    -v "$PWD/models":/models:ro \
    -v "$PWD/ZOOM0020_LR.wav":/input/ZOOM0020_LR.wav:ro \
    -v "$PWD/output":/output:z \
    whisperx-local \
    /input/ZOOM0020_LR.wav \
    --model KBLab/kb-whisper-large \
    --device cpu --compute_type float32 \
    --diarize --output_dir /output --output_format txt
```

The resulting `/output/ZOOM0020_LR.txt` (or `.srt`, `.json`, etc.) will
contain the speaker‑labelled transcription.  Adjust the mounted paths as needed for your environment.

For a higher-level interface, use `manage.py` (on the host) and `transcribe.py` instead of
invoking the container directly.
