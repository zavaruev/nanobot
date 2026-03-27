"""Voice transcription provider using Groq."""

import os
from pathlib import Path

import httpx
from loguru import logger


class GroqTranscriptionProvider:
    """
    Voice transcription provider using Groq's Whisper API.

    Groq offers extremely fast transcription with a generous free tier.
    """

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.api_url = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe an audio file using Groq.

        Args:
            file_path: Path to the audio file.

        Returns:
            Transcribed text.
        """
        if not self.api_key:
            logger.warning("Groq API key not configured for transcription")
            return ""

        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        try:
            async with httpx.AsyncClient() as client:
                with open(path, "rb") as f:
                    files = {
                        "file": (path.name, f),
                        "model": (None, "whisper-large-v3"),
                    }
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                    }

                    response = await client.post(
                        self.api_url,
                        headers=headers,
                        files=files,
                        timeout=60.0
                    )

                    response.raise_for_status()
                    data = response.json()
                    return data.get("text", "")

        except Exception as e:
            logger.error("Groq transcription error: {}", e)
            return ""


class WhisperTranscriber:
    """
    Local Whisper transcription provider using a self-hosted whisper-server
    (Speaches-AI compatible OpenAI API).
    """

    def __init__(self, base_url: str = "http://192.168.22.102:8000", model: str = "large-v3"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def transcribe(self, file_path: str | Path) -> str:
        """
        Transcribe an audio file using local whisper-server.

        Args:
            file_path: Path to the audio file (WAV).

        Returns:
            Transcribed text.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error("Audio file not found: {}", file_path)
            return ""

        try:
            async with httpx.AsyncClient() as client:
                with open(path, "rb") as f:
                    files = {
                        "file": (path.name, f, "audio/wav"),
                    }
                    data = {
                        "model": self.model,
                        "language": "ru",
                        "response_format": "json",
                    }
                    response = await client.post(
                        f"{self.base_url}/v1/audio/transcriptions",
                        files=files,
                        data=data,
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    result = response.json()
                    return result.get("text", "")

        except Exception as e:
            logger.error("Whisper transcription error: {}", e)
            return ""
