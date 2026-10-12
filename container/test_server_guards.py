"""Runnable checks for the request-validation guards in server.py.

Plain script, stdlib only — run it from anywhere:

    python3 container/test_server_guards.py

heavy deps (torch / torchaudio / fastapi / whisperx) are stubbed so the guards
can be exercised without a model, a GPU or a container.  Every assert here
corresponds to a behaviour the endpoints promise in README.md.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

HERE = Path(__file__).parent


class HTTPException(Exception):
    def __init__(self, status_code: int = 500, detail=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _stub(name: str, **attrs) -> types.ModuleType:
    mod = MagicMock(name=name)
    mod.__name__ = name
    mod.__dict__.update(attrs)
    sys.modules[name] = mod
    return mod


def load_server():
    """Import container/server.py with all heavy deps stubbed."""
    routes = lambda *a, **k: (lambda fn: fn)  # noqa: E731
    _stub(
        "fastapi",
        FastAPI=lambda *a, **k: types.SimpleNamespace(get=routes, post=routes),
        File=lambda *a, **k: None,
        Form=lambda *a, **k: None,
        HTTPException=HTTPException,
        Request=object,
        UploadFile=object,
    )
    _stub("fastapi.responses", JSONResponse=object)
    _stub("numpy", arange=lambda *a, **k: [0.0], floating=float, integer=int, ndarray=object)
    _stub(
        "torch",
        set_num_threads=lambda n: None,
        cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
    )

    # One torchaudio pipeline whose weights sit FLAT in the cache while its
    # _path is nested — exactly the layout torch.hub produces.
    cache = Path("/tmp/server-guard-torch/hub/checkpoints")
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "fr_weight.pt").write_bytes(b"x")
    os.environ["TORCH_HOME"] = str(cache.parent.parent)
    _stub(
        "torchaudio",
        pipelines=types.SimpleNamespace(
            __all__=["VOXPOPULI_ASR_BASE_10K_FR", "WAV2VEC2_ASR_BASE_960H"],
            VOXPOPULI_ASR_BASE_10K_FR=types.SimpleNamespace(_path="pt/voxpopuli/asr/10k/fr/fr_weight.pt"),
            WAV2VEC2_ASR_BASE_960H=types.SimpleNamespace(_path="absent_pipeline.pth"),
        ),
    )

    _stub("whisperx")
    _stub(
        "whisperx.alignment",
        DEFAULT_ALIGN_MODELS_TORCH={"en": "WAV2VEC2_ASR_BASE_960H", "fr": "VOXPOPULI_ASR_BASE_10K_FR"},
        DEFAULT_ALIGN_MODELS_HF={"sv": "KBLab/wav2vec2-large-voxrex-swedish"},
    )
    for name in ("whisperx.asr", "whisperx.audio", "whisperx.diarize", "whisperx.utils"):
        _stub(name)

    spec = importlib.util.spec_from_file_location("server", HERE / "server.py")
    server = importlib.util.module_from_spec(spec)
    sys.modules["server"] = server
    spec.loader.exec_module(server)
    server._config = server.ModelConfig()
    server._ready = True
    server._load_models = lambda cfg: setattr(server, "_config", cfg)  # no real loading
    server._unload_models = lambda: None
    return server


srv = load_server()
REQ = types.SimpleNamespace(headers={})
NOPE = object()


def status(coro) -> tuple[int, object]:
    """(status, detail) an endpoint would return; 200 if it completes."""
    try:
        body = asyncio.run(coro)
    except HTTPException as exc:
        return exc.status_code, exc.detail
    except Exception as exc:  # an uncaught error is a 500 for the client
        return 500, f"{type(exc).__name__}: {exc}"
    return 200, body


def expect(want: int, label: str, code: int, detail=NOPE):
    assert code == want, f"{label}: expected {want}, got {code} ({detail})"


def reload_(body: dict) -> tuple[int, object]:
    return status(srv.reload_models(dict(body)))


def transcribe(params: str) -> tuple[int, object]:
    return status(srv.transcribe(request=REQ, audio=None, params=params))


# ── A: a torchaudio weight seeded flat must not be reported as missing ───────
expect(True, "nested _path, flat weight", srv._torch_pipeline_cached("VOXPOPULI_ASR_BASE_10K_FR"))
expect(
    False,
    "genuinely absent pipeline still reported missing",
    srv._torch_pipeline_cached("WAV2VEC2_ASR_BASE_960H"),
)

# ── D: the mount rule, in one place, for every shape of value ────────────────
for good in ["/models/extra/kb-whisper-large-ct2", "//models/extra/x", "KBLab/kb-whisper-large"]:
    expect(None, f"accept {good}", srv._model_path_error(good))
for bad in [
    "/models/extra",
    "/models/extraX/y",
    "/etc/passwd",
    "/models/extra/../etc",
    "../outside",
    "~/torch/hub",
]:
    expect(str, "reject " + bad, type(srv._model_path_error(bad)))

# ── B: /reload and /transcribe share one model-field rule ────────────────────
expect(str, "model must be a string", type(srv._model_field_error("model", 123)))
expect(str, "empty model", type(srv._model_field_error("model", "")))
expect(str, "model may not be null", type(srv._model_field_error("model", None)))
expect(None, "align_model may be null (back to auto-select)", srv._model_field_error("align_model", None))
expect(str, "out-of-mount path", type(srv._model_field_error("diarize_model", "/etc/x")))
expect(None, "in-mount path", srv._model_field_error("align_model", "/models/extra/wav2vec2"))

code, detail = reload_({"model": 123})
expect(400, "reload: non-string model is a client error", code, detail)
code, detail = reload_({"model": None})
expect(400, "reload: null model must not unload everything", code, detail)
expect(True, "server still ready after the rejected reload", srv._ready)

code, detail = reload_({"diarize_model": None})
expect(200, "reload: null diarize_model means the env default", code, detail)
expect(
    srv._DEFAULT_DIARIZE_MODEL,
    "null diarize_model resolves to the env default, not None",
    srv._config.diarize_model,
)

# A packages.json bundle must not smuggle an out-of-mount path past the check.
srv._load_packages = lambda: {"sneaky": {"model": "/etc/passwd"}}
code, detail = reload_({"package": "sneaky"})
expect(400, "reload: package-supplied path is checked too", code, detail)

srv._load_packages = lambda: {"nope": {}}
code, detail = reload_({"package": "does-not-exist"})
expect(404, "unknown package stays a 404 (not swallowed into 500)", code, detail)

# ── C: /transcribe params are typed against the documented schema ────────────
for bad in [
    "null",
    "[1,2]",
    '"a string"',
    '{"language": 5, "no_align": true}',
    '{"language": {"x": 1}}',
    '{"batch_size": "8"}',
    '{"chunk_size": null}',
    '{"diarize": "false"}',
    '{"min_speakers": "2"}',
    '{"task": "transkribe"}',
    '{"interpolate_method": "sideways"}',
    '{"output_format": ["nope"]}',
    '{"output_format": "srt,"}',
    '{"align_model": 5}',
    '{"align_model": "/etc/passwd"}',
    '{"diarize_model": "../outside"}',
]:
    code, detail = transcribe(bad)
    expect(422, f"reject {bad}", code, detail)

# Well-formed params must still pass the same validator (no false rejections).
srv._load_packages = lambda: {
    "en-standard": json.loads(Path(HERE.parent / "models/packages.json").read_text())["en-standard"]
}
code, detail = reload_({"package": "en-standard"})
expect(200, "the repo's own en-standard package still reloads", code, detail)
expect(None, "its align_model: null means auto-select", srv._config.align_model)

for ok in [
    '{"language": "sv", "batch_size": 8, "diarize": true}',
    '{"language": null, "min_speakers": null, "output_format": ["srt", "txt"]}',
    '{"align_model": null, "diarize_model": null, "verbose": true}',
]:
    code, detail = transcribe(ok)
    assert not (code == 422 and "must be" in str(detail)), f"{ok}: wrongly rejected ({detail})"

# "all" inside a list means every format, not none of them.
expect(srv.ALL_FORMATS, '"all" in a list', srv._select_formats(["srt", "all"]))
expect(["srt"], "single format", srv._select_formats("srt"))
expect(srv.ALL_FORMATS, '"all" alone', srv._select_formats("all"))

print("all guard checks passed")
