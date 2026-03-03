#!/usr/bin/env python3
"""
WhisperX HTTP-over-Unix-Domain-Socket server.

Runs *inside* the container.  Listens on a Unix domain socket at
/run/api/whisperx.sock – the parent directory is bind-mounted from the
host so the host can connect without any TCP networking.

Endpoints
---------
GET  /health       → model status and config
GET  /models       → inventory of models available in the mounted caches
GET  /params       → full parameter schema for /transcribe and /reload
POST /transcribe   → multipart: audio=<binary>, params=<JSON>
POST /reload       → JSON body – reload model with new config

Environment variables (all optional – see ModelConfig defaults)
----------------------------------------------------------------
WHISPERX_MODEL               Whisper model name or path  (small)
WHISPERX_DEVICE              cpu | cuda                  (cpu)
WHISPERX_DEVICE_INDEX        int                         (0)
WHISPERX_COMPUTE_TYPE        default | float16 | float32 | int8  (default)
WHISPERX_LANGUAGE            ISO-639-1 code or None      (None = auto-detect)
WHISPERX_BATCH_SIZE          int                         (8)
WHISPERX_THREADS             int                         (4)
WHISPERX_MODEL_DIR           custom download root        (None)
HF_TOKEN                     HuggingFace token           (None)
WHISPERX_BEAM_SIZE           int                         (5)
WHISPERX_BEST_OF             int                         (10)
WHISPERX_PATIENCE            float                       (1.0)
WHISPERX_LENGTH_PENALTY      float                       (1.0)
WHISPERX_TEMPERATURE         float                       (0.0)
WHISPERX_TEMP_INCREMENT      float or None               (0.2)
WHISPERX_COMPRESSION_THR     float                       (2.4)
WHISPERX_LOGPROB_THR         float                       (-1.0)
WHISPERX_NO_SPEECH_THR       float                       (0.6)
WHISPERX_SUPPRESS_TOKENS     comma-separated ints        (-1)
WHISPERX_SUPPRESS_NUMERALS   true | false                (false)
WHISPERX_CONDITION_ON_PREV   true | false                (false)
WHISPERX_INITIAL_PROMPT      string                      (None)
WHISPERX_HOTWORDS            string                      (None)
WHISPERX_VAD_METHOD          pyannote | silero           (pyannote)
WHISPERX_VAD_ONSET           float                       (0.500)
WHISPERX_VAD_OFFSET          float                       (0.363)
WHISPERX_CHUNK_SIZE          int                         (30)
OFFLINE                      1 | 0  – enforce HF offline mode  (1)
"""

import asyncio
import dataclasses
import gc
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

# ── configure caches before any whisperx import ──────────────────────────────
# /models/hf   → the HF hub cache (host ~/.cache/huggingface/hub mounted here)
# /models/extra → project models dir (torch cache, alignment models, etc.)
os.environ.setdefault("HF_HOME", "/models/hf")
os.environ.setdefault("HF_HUB_CACHE", "/models/hf")
os.environ.setdefault("XDG_CACHE_HOME", "/models/extra/cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TORCH_HOME", "/models/extra/cache/torch")
if os.environ.get("OFFLINE", "1") != "0":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from whisperx.asr import load_model as _load_asr_model
from whisperx.audio import load_audio
from whisperx.alignment import align, load_align_model
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from whisperx.utils import (
    WriteAudacity,
    WriteJSON,
    WriteSRT,
    WriteTSV,
    WriteTXT,
    WriteVTT,
)

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("whisperx.server")

# ── constants ─────────────────────────────────────────────────────────────────
SOCKET_PATH = "/run/api/whisperx.sock"

WRITER_CLASSES = {
    "txt": WriteTXT,
    "srt": WriteSRT,
    "vtt": WriteVTT,
    "tsv": WriteTSV,
    "json": WriteJSON,
    "aud": WriteAudacity,
}
ALL_FORMATS = list(WRITER_CLASSES.keys())


# ── model configuration ───────────────────────────────────────────────────────

def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))

def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))

def _env_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).lower() in ("1", "true", "yes")

def _env_opt_str(key: str) -> Optional[str]:
    v = os.environ.get(key, "")
    return v if v else None


@dataclass
class ModelConfig:
    # ── model identity ──────────────────────────────────────────────────────
    model: str                        = field(default_factory=lambda: os.environ.get("WHISPERX_MODEL", "small"))
    device: str                       = field(default_factory=lambda: os.environ.get("WHISPERX_DEVICE", "cpu"))
    device_index: int                 = field(default_factory=lambda: _env_int("WHISPERX_DEVICE_INDEX", 0))
    compute_type: str                 = field(default_factory=lambda: os.environ.get("WHISPERX_COMPUTE_TYPE", "default"))
    language: Optional[str]           = field(default_factory=lambda: _env_opt_str("WHISPERX_LANGUAGE"))
    batch_size: int                   = field(default_factory=lambda: _env_int("WHISPERX_BATCH_SIZE", 8))
    threads: int                      = field(default_factory=lambda: _env_int("WHISPERX_THREADS", 4))
    model_dir: Optional[str]          = field(default_factory=lambda: _env_opt_str("WHISPERX_MODEL_DIR"))
    hf_token: Optional[str]           = field(default_factory=lambda: _env_opt_str("HF_TOKEN"))
    # ── ASR / decoding options (baked in at load_model time) ────────────────
    beam_size: int                    = field(default_factory=lambda: _env_int("WHISPERX_BEAM_SIZE", 5))
    best_of: int                      = field(default_factory=lambda: _env_int("WHISPERX_BEST_OF", 10))
    patience: float                   = field(default_factory=lambda: _env_float("WHISPERX_PATIENCE", 1.0))
    length_penalty: float             = field(default_factory=lambda: _env_float("WHISPERX_LENGTH_PENALTY", 1.0))
    temperature: float                = field(default_factory=lambda: _env_float("WHISPERX_TEMPERATURE", 0.0))
    temperature_increment_on_fallback: Optional[float] = field(
        default_factory=lambda: _env_float("WHISPERX_TEMP_INCREMENT", 0.2)
    )
    compression_ratio_threshold: float = field(default_factory=lambda: _env_float("WHISPERX_COMPRESSION_THR", 2.4))
    logprob_threshold: float           = field(default_factory=lambda: _env_float("WHISPERX_LOGPROB_THR", -1.0))
    no_speech_threshold: float         = field(default_factory=lambda: _env_float("WHISPERX_NO_SPEECH_THR", 0.6))
    suppress_tokens: str               = field(default_factory=lambda: os.environ.get("WHISPERX_SUPPRESS_TOKENS", "-1"))
    suppress_numerals: bool            = field(default_factory=lambda: _env_bool("WHISPERX_SUPPRESS_NUMERALS", False))
    condition_on_previous_text: bool   = field(default_factory=lambda: _env_bool("WHISPERX_CONDITION_ON_PREV", False))
    initial_prompt: Optional[str]      = field(default_factory=lambda: _env_opt_str("WHISPERX_INITIAL_PROMPT"))
    hotwords: Optional[str]            = field(default_factory=lambda: _env_opt_str("WHISPERX_HOTWORDS"))
    # ── VAD options (baked in at load_model time) ───────────────────────────
    vad_method: str    = field(default_factory=lambda: os.environ.get("WHISPERX_VAD_METHOD", "pyannote"))
    vad_onset: float   = field(default_factory=lambda: _env_float("WHISPERX_VAD_ONSET", 0.500))
    vad_offset: float  = field(default_factory=lambda: _env_float("WHISPERX_VAD_OFFSET", 0.363))
    chunk_size: int    = field(default_factory=lambda: _env_int("WHISPERX_CHUNK_SIZE", 30))


# ── server state ──────────────────────────────────────────────────────────────

_config: Optional[ModelConfig] = None
_asr_model: Any = None
_align_models: dict[str, tuple] = {}   # language → (model, metadata)
_diarize_pipelines: dict[str, Any] = {}  # model_name → DiarizationPipeline
_ready: bool = False
_reloading: bool = False
_lock = asyncio.Lock()


# ── model lifecycle ───────────────────────────────────────────────────────────

def _build_asr_options(cfg: ModelConfig) -> dict:
    temp = cfg.temperature
    if cfg.temperature_increment_on_fallback is not None:
        temp = tuple(np.arange(temp, 1.0 + 1e-6, cfg.temperature_increment_on_fallback))
    else:
        temp = [temp]
    return {
        "beam_size": cfg.beam_size,
        "best_of": cfg.best_of,
        "patience": cfg.patience,
        "length_penalty": cfg.length_penalty,
        "temperatures": temp,
        "compression_ratio_threshold": cfg.compression_ratio_threshold,
        "log_prob_threshold": cfg.logprob_threshold,
        "no_speech_threshold": cfg.no_speech_threshold,
        "condition_on_previous_text": cfg.condition_on_previous_text,
        "initial_prompt": cfg.initial_prompt,
        "hotwords": cfg.hotwords,
        "suppress_tokens": [int(x) for x in cfg.suppress_tokens.split(",")],
        "suppress_numerals": cfg.suppress_numerals,
    }


def _load_models(cfg: ModelConfig) -> None:
    global _config, _asr_model, _align_models, _diarize_pipelines, _ready
    if cfg.threads > 0:
        torch.set_num_threads(cfg.threads)
    logger.info("Loading ASR model %s on %s (%s) …", cfg.model, cfg.device, cfg.compute_type)
    t0 = time.time()
    asr = _load_asr_model(
        cfg.model,
        device=cfg.device,
        device_index=cfg.device_index,
        download_root=cfg.model_dir,
        compute_type=cfg.compute_type,
        language=cfg.language,
        asr_options=_build_asr_options(cfg),
        vad_method=cfg.vad_method,
        vad_options={
            "chunk_size": cfg.chunk_size,
            "vad_onset": cfg.vad_onset,
            "vad_offset": cfg.vad_offset,
        },
        local_files_only=True,
        threads=max(cfg.threads, 4),
        use_auth_token=cfg.hf_token,
    )
    logger.info("ASR model ready in %.1fs", time.time() - t0)
    _config = cfg
    _asr_model = asr
    _align_models = {}          # invalidate cached align models on reload
    _diarize_pipelines = {}     # invalidate cached diarize pipelines on reload
    _ready = True


def _unload_models() -> None:
    global _asr_model, _align_models, _diarize_pipelines, _ready
    _ready = False
    if _asr_model is not None:
        del _asr_model
        _asr_model = None
    for _, (m, _) in list(_align_models.items()):
        del m
    _align_models = {}
    for _, p in list(_diarize_pipelines.items()):
        del p
    _diarize_pipelines = {}
    gc.collect()
    torch.cuda.empty_cache()


def _get_align_model(language: str, cfg: ModelConfig):
    if language not in _align_models:
        logger.info("Loading alignment model for '%s' …", language)
        m, meta = load_align_model(
            language, cfg.device,
            model_dir=cfg.model_dir,
            model_cache_only=True,
        )
        _align_models[language] = (m, meta)
    return _align_models[language]


def _get_diarize_pipeline(model_name: str, cfg: ModelConfig):
    if model_name not in _diarize_pipelines:
        logger.info("Loading diarization pipeline '%s' …", model_name)
        _diarize_pipelines[model_name] = DiarizationPipeline(
            model_name=model_name,
            token=cfg.hf_token,
            device=cfg.device,
            cache_dir=cfg.model_dir,
        )
    return _diarize_pipelines[model_name]


# ── output formatting ─────────────────────────────────────────────────────────

def _to_python(obj):
    """Recursively convert numpy scalars/arrays to plain Python types."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_python(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _format_outputs(result: dict, formats: list[str], writer_args: dict) -> dict[str, str]:
    """Render whisperx result dict into text for each requested format."""
    outputs: dict[str, str] = {}
    for fmt in formats:
        cls = WRITER_CLASSES.get(fmt)
        if cls is None:
            logger.warning("Unknown output format '%s', skipping", fmt)
            continue
        buf = StringIO()
        # writer_instance.write_result writes directly to the TextIO we pass
        cls("/tmp").write_result(result, buf, writer_args)
        outputs[fmt] = buf.getvalue()
    return outputs


# ── FastAPI app ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_models(ModelConfig())
    yield
    _unload_models()


app = FastAPI(title="WhisperX API", version="1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "ready": _ready,
        "reloading": _reloading,
        "model": _config.model if _config else None,
        "device": _config.device if _config else None,
        "compute_type": _config.compute_type if _config else None,
        "language": _config.language if _config else None,
        "vad_method": _config.vad_method if _config else None,
        "align_models_cached": list(_align_models.keys()),
        "diarize_pipelines_cached": list(_diarize_pipelines.keys()),
    }


# ── model inventory (used by /models) ────────────────────────────────────────

# Static metadata for models that may be present in the caches.
# Keys are HuggingFace model IDs exactly as you would pass to --model.
# role:
#   "asr"         – can be passed as the 'model' field to /reload or manage.py start
#   "alignment"   – used automatically by whisperx for forced word alignment
#   "diarization" – can be passed as 'diarize_model' in /transcribe params
#   "vad"         – used internally by the VAD pipeline (not user-selectable directly)
#   "embedding"   – used internally by the diarization pipeline
_MODEL_METADATA: dict[str, dict] = {
    # ── ASR models ──────────────────────────────────────────────────────────
    "KBLab/kb-whisper-large": {
        "role": "asr",
        "architecture": "faster-whisper",
        "languages": ["sv"],
        "description": (
            "Swedish-optimised Whisper large model fine-tuned by KBLab (National Library "
            "of Sweden).  Best accuracy for Swedish speech.  Use compute_type=float32 on "
            "CPU or float16 on GPU."
        ),
    },
    "openai/whisper-large-v2": {
        "role": "asr",
        "architecture": "faster-whisper",
        "languages": ["multilingual"],
        "description": (
            "OpenAI Whisper large-v2.  Strong general multilingual model.  "
            "Use Systran/faster-whisper-large-v2 if you have that cached instead."
        ),
    },
    "openai/whisper-large-v3": {
        "role": "asr",
        "architecture": "faster-whisper",
        "languages": ["multilingual"],
        "description": "OpenAI Whisper large-v3.  Latest general multilingual model from OpenAI.",
    },
    "openai/whisper-tiny": {
        "role": "asr",
        "architecture": "faster-whisper",
        "languages": ["multilingual"],
        "description": "Whisper tiny — extremely fast but low accuracy.  Useful for testing.",
    },
    "Systran/faster-whisper-small": {
        "role": "asr",
        "architecture": "faster-whisper",
        "languages": ["multilingual"],
        "description": (
            "Faster-whisper small model in CTranslate2 format.  "
            "Good balance of speed and accuracy for short clips."
        ),
    },
    # ── alignment models ────────────────────────────────────────────────────
    "KBLab/wav2vec2-large-voxrex-swedish": {
        "role": "alignment",
        "languages": ["sv"],
        "description": (
            "Swedish wav2vec2 model used for forced word-level alignment.  "
            "Selected automatically when language='sv'."
        ),
    },
    "viktor-enzell/wav2vec2-large-voxrex-swedish-4gram": {
        "role": "alignment",
        "languages": ["sv"],
        "description": "Alternative Swedish wav2vec2 alignment model with 4-gram LM.",
    },
    # ── diarization / VAD ───────────────────────────────────────────────────
    "pyannote/speaker-diarization-community-1": {
        "role": "diarization",
        "description": (
            "Pyannote community diarization pipeline.  Pass as 'diarize_model' in "
            "/transcribe params, or use the default by setting diarize=true."
        ),
    },
    "pyannote/segmentation": {
        "role": "vad",
        "description": (
            "Pyannote segmentation model used as the VAD backbone when "
            "vad_method='pyannote' (the default)."
        ),
    },
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "role": "embedding",
        "description": (
            "Multilingual sentence embedding model used by the pyannote diarization "
            "pipeline for speaker representation."
        ),
    },
}


def _scan_hf_cache(hf_cache_dir: str) -> list[dict]:
    """
    Scan the HuggingFace hub cache directory and return one entry per model found.

    Each entry:
      model_id      – the HF model ID (e.g. 'KBLab/kb-whisper-large')
      cache_path    – absolute path to the cache directory for this model
      role          – from _MODEL_METADATA, or 'unknown' if not recognised
      description   – from _MODEL_METADATA, or None
      loaded        – True if this model is currently loaded as the active ASR model
      metadata      – any additional fields from _MODEL_METADATA
    """
    results = []
    try:
        cache = Path(hf_cache_dir)
        if not cache.is_dir():
            return results
        for entry in sorted(cache.iterdir()):
            name = entry.name
            # HF hub entries are named  models--{org}--{repo}
            if not name.startswith("models--"):
                continue
            # Reconstruct the model ID: 'models--KBLab--kb-whisper-large' → 'KBLab/kb-whisper-large'
            parts = name[len("models--"):].split("--", 1)
            if len(parts) != 2:
                continue
            model_id = f"{parts[0]}/{parts[1]}"
            meta = _MODEL_METADATA.get(model_id, {})
            record: dict = {
                "model_id": model_id,
                "cache_path": str(entry),
                "role": meta.get("role", "unknown"),
                "loaded": bool(_config and _config.model == model_id),
            }
            if "description" in meta:
                record["description"] = meta["description"]
            if "languages" in meta:
                record["languages"] = meta["languages"]
            if "architecture" in meta:
                record["architecture"] = meta["architecture"]
            results.append(record)
    except Exception as exc:
        logger.warning("Could not scan HF cache at %s: %s", hf_cache_dir, exc)
    return results


@app.get("/models")
async def models():
    """
    Report which models are available in the mounted model caches.

    Scans the HuggingFace hub cache (mounted at /models/hf) and classifies
    each entry by its role in the whisperx pipeline:

      asr          Models that can be passed as 'model' to POST /reload
                   or manage.py start --model.
      alignment    Forced-alignment models selected automatically by language.
      diarization  Speaker diarization pipelines selectable via the
                   'diarize_model' transcribe param.
      vad          Voice Activity Detection backbone (internal, not user-selectable).
      embedding    Speaker embedding model (internal, used by diarization).
      unknown      Present in the cache but not in the known-model registry.

    Response fields
    ---------------
    available       Full list of all cached models with role and metadata.
    by_role         The same list grouped by role for easy filtering.
    currently_loaded
        Which models are actively loaded in memory right now (ASR model
        plus any lazily-loaded alignment / diarization pipelines).
    """
    hf_cache = os.environ.get("HF_HUB_CACHE", "/models/hf")
    all_models = _scan_hf_cache(hf_cache)

    # Group by role
    by_role: dict[str, list] = {}
    for m in all_models:
        by_role.setdefault(m["role"], []).append(m)

    return {
        "available": all_models,
        "by_role": by_role,
        "currently_loaded": {
            "asr": _config.model if _config else None,
            "alignment": list(_align_models.keys()),
            "diarization": list(_diarize_pipelines.keys()),
        },
    }


# ── parameter schema (static, used by /params) ────────────────────────────────

# Per-request parameters accepted by POST /transcribe (the `params` JSON field).
# These are intentionally NOT part of ModelConfig – they are evaluated fresh on
# every request, so no reload is needed when they change.
_TRANSCRIBE_PARAMS: dict = {
    "language": {
        "type": "string | null",
        "default": None,
        "description": (
            "ISO-639-1 language code (e.g. 'sv', 'en', 'de'). "
            "Pass null or omit to enable automatic language detection. "
            "Overrides the server-level default set at startup."
        ),
    },
    "task": {
        "type": "string",
        "default": "transcribe",
        "enum": ["transcribe", "translate"],
        "description": (
            "'transcribe' returns text in the source language. "
            "'translate' forces English output regardless of input language. "
            "Forced alignment is skipped when task='translate'."
        ),
    },
    "batch_size": {
        "type": "int",
        "default": 8,
        "description": "Number of audio chunks processed in parallel during ASR inference. "
                       "Lower values reduce peak memory use; higher values may improve speed on GPU.",
    },
    "chunk_size": {
        "type": "int",
        "default": 30,
        "description": "VAD chunk length in seconds. Longer chunks give the model more context "
                       "but increase latency and memory use.",
    },
    "no_align": {
        "type": "bool",
        "default": False,
        "description": "Skip forced alignment. Segments will have sentence-level start/end "
                       "timestamps only; the 'words' array will be absent.",
    },
    "align_model": {
        "type": "string | null",
        "default": None,
        "description": "HuggingFace model ID to use for forced alignment. "
                       "Defaults to the best available model for the detected language.",
    },
    "interpolate_method": {
        "type": "string",
        "default": "nearest",
        "enum": ["nearest", "linear", "ignore"],
        "description": "How to assign timestamps to words that fall in silent gaps between "
                       "aligned segments.",
    },
    "return_char_alignments": {
        "type": "bool",
        "default": False,
        "description": "Include character-level alignment timestamps in each word entry.",
    },
    "diarize": {
        "type": "bool",
        "default": False,
        "description": "Run speaker diarization after transcription. Each segment will gain a "
                       "'speaker' field (e.g. 'SPEAKER_00'). Requires the pyannote diarization "
                       "model to be available in the model cache.",
    },
    "min_speakers": {
        "type": "int | null",
        "default": None,
        "description": "Minimum number of speakers to assume during diarization. "
                       "Only used when diarize=true.",
    },
    "max_speakers": {
        "type": "int | null",
        "default": None,
        "description": "Maximum number of speakers to assume during diarization. "
                       "Only used when diarize=true.",
    },
    "diarize_model": {
        "type": "string",
        "default": "pyannote/speaker-diarization-community-1",
        "description": "HuggingFace model ID for the diarization pipeline. "
                       "Only used when diarize=true.",
    },
    "speaker_embeddings": {
        "type": "bool",
        "default": False,
        "description": "Return per-speaker embedding vectors alongside segment annotations. "
                       "Only used when diarize=true.",
    },
    "output_format": {
        "type": "string | list[string]",
        "default": "all",
        "enum": ["all", "txt", "srt", "vtt", "tsv", "json", "aud"],
        "description": (
            "Which output format(s) to include in the response 'outputs' dict. "
            "'all' returns every format. "
            "Pass a list (e.g. ['srt','txt']) for multiple specific formats. "
            "Formats: txt=plain text, srt=SubRip subtitles, vtt=WebVTT subtitles, "
            "tsv=tab-separated with timestamps, json=full segment JSON, "
            "aud=Audacity label track."
        ),
    },
    "highlight_words": {
        "type": "bool",
        "default": False,
        "description": "Underline each word at the moment it is spoken in SRT/VTT output. "
                       "Requires word-level alignment (no_align must be false).",
    },
    "max_line_width": {
        "type": "int | null",
        "default": None,
        "description": "Wrap subtitle lines at this many characters. Null = no wrapping.",
    },
    "max_line_count": {
        "type": "int | null",
        "default": None,
        "description": "Maximum number of lines per subtitle block. Null = no limit.",
    },
    "verbose": {
        "type": "bool",
        "default": False,
        "description": "Enable verbose logging inside the container during transcription.",
    },
    "print_progress": {
        "type": "bool",
        "default": False,
        "description": "Log segment-by-segment progress to the container stdout.",
    },
}


def _model_config_schema() -> dict:
    """
    Build a parameter schema for POST /reload by introspecting ModelConfig.

    Returns a dict of field_name → {type, default, env_var, description} derived
    directly from the dataclass so it can never drift out of sync with the code.
    """
    # Human-readable descriptions and env-var names for each ModelConfig field.
    # Fields absent from this map will still appear in the output but without a
    # description or env_var hint.
    _meta: dict[str, dict] = {
        "model":           {"env": "WHISPERX_MODEL",
                            "description": "Whisper model name or HuggingFace repo ID "
                                           "(e.g. 'small', 'large-v2', 'KBLab/kb-whisper-large')."},
        "device":          {"env": "WHISPERX_DEVICE",      "enum": ["cpu", "cuda"],
                            "description": "Compute device. Use 'cpu' unless a CUDA GPU is available."},
        "device_index":    {"env": "WHISPERX_DEVICE_INDEX",
                            "description": "GPU device index when device='cuda'."},
        "compute_type":    {"env": "WHISPERX_COMPUTE_TYPE",
                            "enum": ["default", "float16", "float32", "int8"],
                            "description": "Quantisation precision. 'float32' is safest on CPU; "
                                           "'float16' or 'int8' for faster GPU inference."},
        "language":        {"env": "WHISPERX_LANGUAGE",
                            "description": "ISO-639-1 language code baked in at model-load time. "
                                           "Can also be overridden per-request without a reload."},
        "batch_size":      {"env": "WHISPERX_BATCH_SIZE",
                            "description": "Default batch size used when not overridden per-request."},
        "threads":         {"env": "WHISPERX_THREADS",
                            "description": "Number of CPU threads allocated to PyTorch."},
        "model_dir":       {"env": "WHISPERX_MODEL_DIR",
                            "description": "Custom directory for model downloads. "
                                           "Null = use the default HF/torch cache."},
        "hf_token":        {"env": "HF_TOKEN",
                            "description": "HuggingFace access token for gated models "
                                           "(e.g. pyannote diarization). Not logged."},
        "beam_size":       {"env": "WHISPERX_BEAM_SIZE",
                            "description": "Beam search width. Higher = better accuracy, slower."},
        "best_of":         {"env": "WHISPERX_BEST_OF",
                            "description": "Number of candidates sampled when temperature > 0."},
        "patience":        {"env": "WHISPERX_PATIENCE",
                            "description": "Beam search patience factor."},
        "length_penalty":  {"env": "WHISPERX_LENGTH_PENALTY",
                            "description": "Exponential length penalty applied to beam scores."},
        "temperature":     {"env": "WHISPERX_TEMPERATURE",
                            "description": "Sampling temperature. 0 = greedy / beam search."},
        "temperature_increment_on_fallback": {
                            "env": "WHISPERX_TEMP_INCREMENT",
                            "description": "Temperature step used when the model falls back due to "
                                           "high compression ratio or low log-probability. "
                                           "Null disables fallback."},
        "compression_ratio_threshold": {
                            "env": "WHISPERX_COMPRESSION_THR",
                            "description": "If the gzip compression ratio of the output exceeds "
                                           "this value the segment is considered failed and "
                                           "temperature fallback is triggered."},
        "logprob_threshold":{"env": "WHISPERX_LOGPROB_THR",
                            "description": "Average log-probability threshold below which "
                                           "temperature fallback is triggered."},
        "no_speech_threshold": {
                            "env": "WHISPERX_NO_SPEECH_THR",
                            "description": "If the no-speech probability exceeds this value the "
                                           "segment is treated as silence."},
        "suppress_tokens": {"env": "WHISPERX_SUPPRESS_TOKENS",
                            "description": "Comma-separated list of token IDs to suppress. "
                                           "'-1' suppresses the default set of special tokens."},
        "suppress_numerals": {
                            "env": "WHISPERX_SUPPRESS_NUMERALS",
                            "description": "Replace numeric digits with their spoken-word forms."},
        "condition_on_previous_text": {
                            "env": "WHISPERX_CONDITION_ON_PREV",
                            "description": "Feed the previous segment's text as a prompt for "
                                           "the next segment. Can improve continuity but risks "
                                           "hallucination loops on long files."},
        "initial_prompt":  {"env": "WHISPERX_INITIAL_PROMPT",
                            "description": "Text prepended as context before the first segment. "
                                           "Useful for domain-specific vocabulary."},
        "hotwords":        {"env": "WHISPERX_HOTWORDS",
                            "description": "Space-separated hotwords boosted during decoding."},
        "vad_method":      {"env": "WHISPERX_VAD_METHOD", "enum": ["pyannote", "silero"],
                            "description": "Voice Activity Detection backend. 'pyannote' gives "
                                           "better accuracy; 'silero' is lighter and faster."},
        "vad_onset":       {"env": "WHISPERX_VAD_ONSET",
                            "description": "VAD probability threshold to start a speech segment."},
        "vad_offset":      {"env": "WHISPERX_VAD_OFFSET",
                            "description": "VAD probability threshold to end a speech segment."},
        "chunk_size":      {"env": "WHISPERX_CHUNK_SIZE",
                            "description": "Default VAD chunk length in seconds. "
                                           "Can also be overridden per-request without a reload."},
    }

    schema: dict = {}
    defaults = dataclasses.asdict(ModelConfig())
    for f in dataclasses.fields(ModelConfig):
        entry: dict = {"type": str(f.type) if isinstance(f.type, str) else type(defaults[f.name]).__name__}
        entry["default"] = defaults[f.name]
        meta = _meta.get(f.name, {})
        if "env" in meta:
            entry["env_var"] = meta["env"]
        if "enum" in meta:
            entry["enum"] = meta["enum"]
        if "description" in meta:
            entry["description"] = meta["description"]
        schema[f.name] = entry
    return schema


@app.get("/params")
async def params():
    """
    Describe the parameters accepted by POST /transcribe and POST /reload.

    This endpoint exists so that clients (including automated agents) can
    discover the full API surface without consulting external documentation.

    Response fields
    ---------------
    transcribe_params
        Per-request options sent as the `params` JSON string in the
        /transcribe multipart body.  These take effect immediately on
        every call; no reload is needed.

    reload_params
        Fields accepted by POST /reload.  These are baked into the ASR
        model at load time.  Changing them requires a reload (which
        briefly makes the server unavailable while the model is swapped).
        The current live values are included alongside the defaults so a
        client can show what is currently active.

    output_formats
        The set of strings valid for the `output_format` transcribe param.

    notes
        Human-readable guidance on the distinction between the two param
        sets and other caveats.
    """
    current_values = dataclasses.asdict(_config) if _config else {}
    reload_schema = _model_config_schema()
    # Annotate each reload param with its current live value
    for key, entry in reload_schema.items():
        if key in current_values:
            entry["current_value"] = current_values[key]

    return {
        "transcribe_params": _TRANSCRIBE_PARAMS,
        "reload_params": reload_schema,
        "output_formats": ALL_FORMATS,
        "notes": {
            "transcribe_params_usage": (
                "Send as a JSON-encoded string in the 'params' multipart field of "
                "POST /transcribe.  Example: "
                'params={"language":"sv","diarize":true,"output_format":"srt"}'
            ),
            "reload_params_usage": (
                "Send as a JSON object body to POST /reload.  Include only the fields "
                "you want to change; omitted fields keep their current value.  "
                "The server will be briefly unavailable while the model reloads."
            ),
            "overlap": (
                "Some fields (language, batch_size, chunk_size) exist in both sets. "
                "The reload version sets the server-wide default; the transcribe version "
                "overrides it for a single request only."
            ),
        },
    }


@app.post("/reload")
async def reload_models(body: dict):
    """
    Reload the ASR model with new configuration.

    Send a JSON object with any subset of ModelConfig fields; omitted
    fields keep their current value.  All cached alignment and
    diarization models are also dropped and will be lazily re-loaded.

    Note: language, batch_size, chunk_size, diarize, min/max_speakers,
    output_format etc. are **per-request** params sent to /transcribe and
    do NOT require a reload.  Reload is only needed for params baked into
    the ASR model at load time (model name, beam_size, vad_method, etc.)
    """
    global _reloading
    if _reloading:
        raise HTTPException(503, "Already reloading, please wait")
    async with _lock:
        _reloading = True
        try:
            # Start from current config, apply only the supplied overrides
            if _config is not None:
                current = dataclasses.asdict(_config)
            else:
                current = dataclasses.asdict(ModelConfig())
            valid_fields = {f.name for f in dataclasses.fields(ModelConfig)}
            current.update({k: v for k, v in body.items() if k in valid_fields})
            new_cfg = ModelConfig(**current)
            _unload_models()
            _load_models(new_cfg)
        except Exception as exc:
            logger.exception("Reload failed")
            raise HTTPException(500, f"Reload failed: {exc}") from exc
        finally:
            _reloading = False
    return {
        "status": "reloaded",
        "model": _config.model,
        "device": _config.device,
        "compute_type": _config.compute_type,
    }


@app.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    params: str = Form("{}"),
):
    """
    Transcribe an audio file.

    Multipart fields
    ----------------
    audio   Binary audio file (any format supported by ffmpeg).
    params  Optional JSON string with per-request whisperx options:

        language            ISO-639-1 code or null (overrides server default)
        task                "transcribe" | "translate"  (default: transcribe)
        batch_size          int   (default: server config)
        chunk_size          int   (default: server config)
        no_align            bool  (default: false)
        align_model         str   (default: auto-selected by language)
        interpolate_method  "nearest" | "linear" | "ignore"
        return_char_alignments  bool
        diarize             bool
        min_speakers        int
        max_speakers        int
        diarize_model       str   (default: pyannote/speaker-diarization-community-1)
        speaker_embeddings  bool
        output_format       "all" | "srt" | "vtt" | "txt" | "tsv" | "json" | "aud"
                            or a list of formats, e.g. ["srt", "txt"]
        highlight_words     bool
        max_line_width      int
        max_line_count      int
        verbose             bool
        print_progress      bool

    Response
    --------
    {
      "language": "sv",
      "duration_seconds": 14.2,
      "segments": [ { "start": 0.0, "end": 1.4, "text": "...", "words": [...] } ],
      "outputs": { "srt": "...", "txt": "...", ... }
    }
    """
    if _reloading:
        raise HTTPException(503, "Server is reloading models – try again shortly")
    if not _ready:
        raise HTTPException(503, "Models not ready")

    try:
        p = json.loads(params)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"Invalid JSON in params: {exc}") from exc

    async with _lock:
        cfg = _config
        work_dir = Path(tempfile.mkdtemp(prefix="whisperx_", dir="/tmp"))
        audio_suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
        audio_path = work_dir / f"audio{audio_suffix}"
        t_start = time.time()
        try:
            # ── save uploaded audio ──────────────────────────────────────────
            audio_bytes = await audio.read()
            audio_path.write_bytes(audio_bytes)

            batch_size = p.get("batch_size", cfg.batch_size)
            chunk_size = p.get("chunk_size", cfg.chunk_size)
            language   = p.get("language", cfg.language) or None
            task       = p.get("task", "transcribe")
            verbose    = bool(p.get("verbose", False))
            print_prog = bool(p.get("print_progress", False))

            # ── VAD + ASR ────────────────────────────────────────────────────
            logger.info("Transcribing '%s' (language=%s, task=%s) …", audio.filename, language or "auto", task)
            audio_array = load_audio(str(audio_path))
            result = _asr_model.transcribe(
                audio_array,
                batch_size=batch_size,
                chunk_size=chunk_size,
                language=language,
                task=task,
                print_progress=print_prog,
                verbose=verbose,
            )
            detected_language = result.get("language") or language or "en"
            align_language = language or detected_language

            # ── alignment ────────────────────────────────────────────────────
            no_align = bool(p.get("no_align", False))
            if task == "translate":
                no_align = True

            if not no_align and result.get("segments"):
                align_model, align_meta = _get_align_model(align_language, cfg)
                if align_model is not None:
                    result = align(
                        result["segments"],
                        align_model,
                        align_meta,
                        audio_array,
                        cfg.device,
                        interpolate_method=p.get("interpolate_method", "nearest"),
                        return_char_alignments=bool(p.get("return_char_alignments", False)),
                        print_progress=print_prog,
                    )

            # ── diarization ──────────────────────────────────────────────────
            if p.get("diarize"):
                diarize_model_name = p.get(
                    "diarize_model",
                    "pyannote/speaker-diarization-community-1",
                )
                pipeline = _get_diarize_pipeline(diarize_model_name, cfg)
                return_emb = bool(p.get("speaker_embeddings", False))
                diarize_result = pipeline(
                    str(audio_path),
                    min_speakers=p.get("min_speakers"),
                    max_speakers=p.get("max_speakers"),
                    return_embeddings=return_emb,
                )
                if return_emb:
                    diarize_segs, speaker_emb = diarize_result
                else:
                    diarize_segs, speaker_emb = diarize_result, None
                result = assign_word_speakers(diarize_segs, result, speaker_emb)

            result["language"] = align_language

            # ── convert numpy scalars to plain Python ────────────────────────
            result = _to_python(result)

            # ── format outputs ───────────────────────────────────────────────
            raw_fmt = p.get("output_format", "all")
            if raw_fmt == "all":
                formats = ALL_FORMATS
            elif isinstance(raw_fmt, list):
                formats = raw_fmt
            else:
                formats = [raw_fmt]

            writer_args = {
                "highlight_words":  bool(p.get("highlight_words", False)),
                "max_line_count":   p.get("max_line_count"),
                "max_line_width":   p.get("max_line_width"),
            }
            outputs = _format_outputs(result, formats, writer_args)

            duration = round(time.time() - t_start, 2)
            logger.info("Done in %.1fs, language=%s, %d segments", duration, align_language, len(result.get("segments", [])))

            return JSONResponse({
                "language": align_language,
                "duration_seconds": duration,
                "segments": result.get("segments", []),
                "outputs": outputs,
            })

        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Transcription failed")
            raise HTTPException(500, f"Transcription error: {exc}") from exc
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


# ── entry point ───────────────────────────────────────────────────────────────

def _run_server() -> None:
    import uvicorn

    sock_dir = Path(SOCKET_PATH).parent
    sock_dir.mkdir(parents=True, exist_ok=True)
    # Remove a stale socket from a previous (crashed) run
    Path(SOCKET_PATH).unlink(missing_ok=True)

    # Allow any process (e.g. an nginx sidecar running as a different uid) to
    # connect() to the socket.  connect() on a Unix stream socket requires
    # write permission on the socket file; with the default umask of 0022 the
    # socket would be created as 0755 (others have no write bit) and the nginx
    # user would be refused.  Setting umask=0 lets the kernel create it 0777.
    os.umask(0o000)

    logger.info("Starting WhisperX server on %s", SOCKET_PATH)
    uvicorn.run(app, uds=SOCKET_PATH, log_level="info")


def _run_standalone() -> None:
    """One-shot mode: delegate directly to the whisperx CLI.

    sys.argv must already have the 'standalone' subcommand removed so that
    whisperx sees exactly what it expects, e.g.::

        sys.argv == ['server.py', '/input/audio.wav', '--model', 'large-v2', ...]
    """
    from whisperx.__main__ import cli
    sys.exit(cli() or 0)


if __name__ == "__main__":
    # Dispatch based on first argument:
    #   (no arg) or "server"  →  UDS API server (persistent, model loaded once)
    #   anything else         →  standalone CLI passthrough (one-shot, --rm)
    #
    # The Containerfile sets CMD ["server"] so the default is always server mode.
    # run_whisper_offline.py prepends "standalone" before the audio path.

    mode = sys.argv[1] if len(sys.argv) > 1 else "server"

    if mode == "server":
        # strip the literal "server" arg so uvicorn never sees it
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        _run_server()
    else:
        # standalone: argv[1:] is already the raw whisperx CLI arguments
        # (audio path first, then flags) – hand them straight to the CLI.
        _run_standalone()
