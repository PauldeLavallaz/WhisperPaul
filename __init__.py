# ComfyUI-OpenAI-Transcribe
# Robust AUDIO -> file resolver (handles tensors; arbitrary shapes)
# OpenAI transcription caller. Returns plain text.

import os
import mimetypes
import tempfile
import wave
import requests

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None


def _to_float(val, default=0.0):
    try:
        if val is None:
            return default
        if isinstance(val, (int, float)):
            return float(val)
        s = str(val).strip()
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def _first_present(d: dict, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _to_numpy(a):
    # Convert torch tensor / list-like to numpy without triggering boolean evaluation
    if torch is not None and isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    if np is not None:
        return np.asarray(a)
    raise RuntimeError("numpy es requerido para serializar AUDIO→WAV. Instalá numpy en el venv de ComfyUI.")


def _arr_to_channels_last(arr):
    """
    Normalize any array to shape (C, T) where:
    - T is the longest dimension (assume samples)
    - C is the product of all other dims (batch*channels*...)
    Examples it handles:
      (T,) -> (1, T)
      (C, T) or (T, C)
      (B, C, T), (C, T, B), (T, C, B), etc.
    """
    import numpy as _np

    arr = _to_numpy(arr)
    if arr.ndim == 0:
        raise RuntimeError("AUDIO vacío.")
    if arr.ndim == 1:
        return arr[_np.newaxis, :]

    # pick the axis with the largest size as sample axis
    sample_axis = int(arr.shape.index(max(arr.shape)))
    # move sample axis to the last position
    arr = _np.moveaxis(arr, sample_axis, -1)  # shape: (..., T)

    # flatten all leading dims into channels
    T = arr.shape[-1]
    C = int(_np.prod(arr.shape[:-1])) if arr.ndim > 1 else 1
    arr = arr.reshape(C, T)
    return arr


class OpenAITranscribe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "model": ("STRING", {"multiline": False, "default": "gpt-4o-mini-transcribe"}),
                "language": ("STRING", {"multiline": False, "default": ""}),  # e.g. 'es'
                "temperature": ("STRING", {"multiline": False, "default": "0"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "OpenAI/Audio"

    # --- helpers ---
    def _audio_to_file(self, audio) -> tuple[str, bool]:
        # direct path
        if isinstance(audio, str) and os.path.exists(audio):
            return audio, False

        # dict with a path
        if isinstance(audio, dict):
            for k in ("path", "filepath", "file", "filename", "temp_path"):
                p = audio.get(k)
                if isinstance(p, str) and os.path.exists(p):
                    return p, False

        # raw samples
        samples, sr = None, None
        if isinstance(audio, dict):
            sr = _first_present(audio, ("sample_rate", "sampling_rate", "sr"))
            samples = _first_present(audio, ("samples", "waveform", "audio", "data"))
        elif isinstance(audio, (list, tuple)) and len(audio) == 2:
            samples, sr = audio[0], audio[1]

        if samples is None:
            raise RuntimeError("AUDIO no contiene path ni samples. Pasá un AUDIO válido (con path o samples).")

        arr = _arr_to_channels_last(samples)

        # normalize to int16 PCM
        import numpy as _np
        if arr.dtype.kind == "f":
            arr = _np.clip(arr, -1.0, 1.0)
            pcm = (arr * 32767.0).astype(_np.int16)
        elif arr.dtype != _np.int16:
            pcm = arr.astype(_np.int16)
        else:
            pcm = arr

        interleaved = pcm.T.reshape(-1)

        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(tmp_path, "wb") as w:
            w.setnchannels(int(pcm.shape[0]))
            w.setsampwidth(2)  # 16-bit
            w.setframerate(int(sr or 44100))
            w.writeframes(interleaved.tobytes())

        return tmp_path, True

    # --- main ---
    def run(self, audio, api_key: str, model: str = "gpt-4o-mini-transcribe",
            language: str = "", temperature: str = "0"):
        key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY no seteada. Pasá la API Key en 'api_key' o seteala como variable de entorno.")

        path, should_cleanup = self._audio_to_file(audio)
        url = "https://api.openai.com/v1/audio/transcriptions"
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"

        temp_val = _to_float(temperature, 0.0)

        try:
            with open(path, "rb") as f:
                files = {"file": (os.path.basename(path), f, mime)}
                data = {
                    "model": model,
                    "temperature": str(temp_val),
                    "response_format": "text",
                }
                if language:
                    data["language"] = language

                headers = {"Authorization": f"Bearer {key}"}
                resp = requests.post(url, headers=headers, data=data, files=files, timeout=600)

            if resp.status_code // 100 != 2:
                snippet = resp.text[:600].strip().replace("\\n", " ")
                raise RuntimeError(f"OpenAI HTTP {resp.status_code}: {snippet}")

            text = resp.text or ""
            return (text,)
        finally:
            if should_cleanup:
                try:
                    os.remove(path)
                except Exception:
                    pass


NODE_CLASS_MAPPINGS = {"OpenAITranscribe": OpenAITranscribe}
NODE_DISPLAY_NAME_MAPPINGS = {"OpenAITranscribe": "OpenAI Transcribe (4o mini)"}
