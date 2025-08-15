# ComfyUI-OpenAI-Transcribe
# OpenAI Audio Transcription node for ComfyUI
# Inputs: audio (AUDIO), api_key (STRING), optional: model/language/temperature
# Output: text (STRING) - transcription
#
# Model default: gpt-4o-mini-transcribe
# Endpoint: POST https://api.openai.com/v1/audio/transcriptions

import os
import requests
import mimetypes
import tempfile
import wave

try:
    import numpy as np
except Exception:
    np = None


class OpenAITranscribe:
    """
    ComfyUI node that accepts an AUDIO object and calls OpenAI's transcription API.
    Returns plain text.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "model": ("STRING", {"multiline": False, "default": "gpt-4o-mini-transcribe"}),
                "language": ("STRING", {"multiline": False, "default": ""}),  # e.g. "es"; blank = auto
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "OpenAI/Audio"

    # ---- helpers ----
    def _audio_to_file(self, audio) -> tuple[str, bool]:
        """
        Resolve an AUDIO object into a readable file path.
        Returns (path, should_delete_after_use).
        Strategy:
          1) If it's already a path in common keys, use it.
          2) If it contains raw samples + sample_rate, write a temp WAV.
        """
        # 1) pre-existing path in typical keys
        if isinstance(audio, str) and os.path.exists(audio):
            return audio, False
        if isinstance(audio, dict):
            for k in ("path", "filepath", "file", "filename", "temp_path"):
                p = audio.get(k)
                if isinstance(p, str) and os.path.exists(p):
                    return p, False

        # 2) build WAV from samples
        samples, sr = None, None
        if isinstance(audio, dict):
            sr = audio.get("sample_rate") or audio.get("sampling_rate") or audio.get("sr") or 44100
            samples = audio.get("samples") or audio.get("waveform") or audio.get("audio") or audio.get("data")
        elif isinstance(audio, (list, tuple)) and len(audio) == 2:
            samples, sr = audio[0], audio[1]

        if samples is None:
            raise RuntimeError("AUDIO no contiene path ni samples reconocibles. Pasá un AUDIO válido (con path o samples).")

        if np is None:
            raise RuntimeError("Este nodo requiere numpy para serializar AUDIO→WAV. Instalá numpy en el venv de ComfyUI.")

        import numpy as _np
        arr = _np.asarray(samples)
        # layout & channels
        if arr.ndim == 1:
            channels = 1
            arr = arr[None, :]  # (1, T)
        elif arr.ndim == 2:
            c_first = arr.shape[0] <= arr.shape[1]
            channels = arr.shape[0] if c_first else arr.shape[1]
            if not c_first:
                arr = arr.T  # (C, T)
        else:
            raise RuntimeError(f"Formato de samples inesperado: {arr.shape}")

        # to int16 PCM
        if arr.dtype.kind == "f":
            arr = _np.clip(arr, -1.0, 1.0)
            pcm = (arr * 32767.0).astype(_np.int16)
        elif arr.dtype == _np.int16:
            pcm = arr
        else:
            pcm = arr.astype(_np.int16)

        interleaved = pcm.T.reshape(-1)

        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(tmp_path, "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(2)  # 16-bit
            w.setframerate(int(sr or 44100))
            w.writeframes(interleaved.tobytes())

        return tmp_path, True

    # ---- main ----
    def run(self, audio, api_key: str, model: str = "gpt-4o-mini-transcribe",
            language: str = "", temperature: float = 0.0):
        key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        if not key:
            raise RuntimeError("OPENAI_API_KEY no seteada. Pasá la API Key en 'api_key' o seteala como variable de entorno.")

        path, should_cleanup = self._audio_to_file(audio)
        url = "https://api.openai.com/v1/audio/transcriptions"
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"

        try:
            with open(path, "rb") as f:
                files = {"file": (os.path.basename(path), f, mime)}
                data = {
                    "model": model,
                    "temperature": str(temperature),
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
