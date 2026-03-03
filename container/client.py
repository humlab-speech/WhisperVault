"""
Thin Python client for the WhisperX UDS API.

Intended for programmatic use from host-side scripts.  Requires httpx>=0.23.

    from container.client import WhisperXClient

    with WhisperXClient() as wx:
        health = wx.health()
        result = wx.transcribe(
            "/path/to/audio.wav",
            language="sv",
            diarize=True,
            output_format=["srt", "txt"],
        )
        print(result["outputs"]["srt"])
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

DEFAULT_SOCKET = os.environ.get("WHISPERX_SOCKET_DIR", "/tmp/whisperx-api") + "/whisperx.sock"


class WhisperXClient:
    """
    HTTP client that speaks to the whisperx server over a Unix Domain Socket.

    Parameters
    ----------
    socket_path : str
        Path to the socket file on the host.
    timeout : float
        Default request timeout in seconds (transcription can take minutes).
    """

    def __init__(
        self,
        socket_path: str = DEFAULT_SOCKET,
        timeout: float = 3600.0,
    ) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required:  pip install 'httpx>=0.23'"
            ) from exc
        self._httpx = httpx
        self._socket_path = socket_path
        self._timeout = timeout
        self._client: Any = None

    # ── context manager ───────────────────────────────────────────────────

    def __enter__(self) -> "WhisperXClient":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def connect(self) -> None:
        self._client = self._httpx.Client(
            transport=self._httpx.HTTPTransport(uds=self._socket_path),
            base_url="http://localhost",
            timeout=self._timeout,
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ── public API ────────────────────────────────────────────────────────

    def health(self) -> dict:
        """Return the server's health/status dict."""
        return self._get("/health")

    def reload(self, **kwargs) -> dict:
        """
        Reload the ASR model with new settings.

        Keyword arguments correspond to ModelConfig fields, e.g.::

            wx.reload(model="KBLab/kb-whisper-large", device="cpu")

        Only the fields you pass are changed; others keep their current value.
        """
        return self._post("/reload", json=kwargs)

    def transcribe(
        self,
        audio: str | os.PathLike,
        *,
        language: Optional[str] = None,
        task: str = "transcribe",
        output_format: str | list[str] = "all",
        diarize: bool = False,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        diarize_model: Optional[str] = None,
        no_align: bool = False,
        align_model: Optional[str] = None,
        interpolate_method: str = "nearest",
        return_char_alignments: bool = False,
        speaker_embeddings: bool = False,
        batch_size: Optional[int] = None,
        chunk_size: Optional[int] = None,
        verbose: bool = False,
        print_progress: bool = False,
        highlight_words: bool = False,
        max_line_width: Optional[int] = None,
        max_line_count: Optional[int] = None,
    ) -> dict:
        """
        Transcribe an audio file.

        Parameters
        ----------
        audio
            Path to the audio file.
        language
            ISO-639-1 code.  ``None`` → auto-detect.
        task
            ``"transcribe"`` or ``"translate"``.
        output_format
            One of ``"all"``, ``"srt"``, ``"vtt"``, ``"txt"``, ``"tsv"``,
            ``"json"``, ``"aud"`` – or a list of these.
        diarize
            Run speaker diarization.
        output_format
            Which format(s) to include in ``result["outputs"]``.

        Returns
        -------
        dict
            ``{ "language": str, "duration_seconds": float,
                "segments": [...], "outputs": {"srt": str, ...} }``
        """
        audio_path = Path(audio)
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        params: dict = dict(
            task=task,
            output_format=output_format,
            no_align=no_align,
            interpolate_method=interpolate_method,
            return_char_alignments=return_char_alignments,
            diarize=diarize,
            speaker_embeddings=speaker_embeddings,
            verbose=verbose,
            print_progress=print_progress,
            highlight_words=highlight_words,
        )
        if language is not None:
            params["language"] = language
        if diarize and min_speakers is not None:
            params["min_speakers"] = min_speakers
        if diarize and max_speakers is not None:
            params["max_speakers"] = max_speakers
        if diarize and diarize_model is not None:
            params["diarize_model"] = diarize_model
        if align_model is not None:
            params["align_model"] = align_model
        if batch_size is not None:
            params["batch_size"] = batch_size
        if chunk_size is not None:
            params["chunk_size"] = chunk_size
        if max_line_width is not None:
            params["max_line_width"] = max_line_width
        if max_line_count is not None:
            params["max_line_count"] = max_line_count

        with open(audio_path, "rb") as fh:
            response = self._client.post(
                "/transcribe",
                files={"audio": (audio_path.name, fh)},
                data={"params": json.dumps(params)},
            )
        response.raise_for_status()
        return response.json()

    # ── internal ──────────────────────────────────────────────────────────

    def _get(self, path: str) -> dict:
        if self._client is None:
            self.connect()
        r = self._client.get(path)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, **kwargs) -> dict:
        if self._client is None:
            self.connect()
        r = self._client.post(path, **kwargs)
        r.raise_for_status()
        return r.json()
