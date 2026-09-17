import httpx
import json
from typing import TypedDict, Literal
from app.core.config import settings

class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str

class OllamaClient:
    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model

    def chat(self, messages: list[Message]):
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": self.model,
            "messages": messages,
            "think": False,
            "stream": False
        }

        try:
            response = httpx.post(
                url,
                json=payload,
                timeout=120.0
            )

            response.raise_for_status()

            data = response.json()

            content = data.get("message", {}).get("content", "")

            if not content:
                raise RuntimeError(
                    "Ollama response không chứa message.content"
                )

            return content

        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Không thể kết nối tới Ollama tại {self.base_url}. "
                "Hãy kiểm tra Ollama server"
            ) from exc
        except httpx.TimeoutException as exc:
            raise RuntimeError(
                "Ollama phản hồi quá lâu và request đã timeout."
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama trả về HTTP {exc.response.status_code}: "
                f"{exc.response.text}"
            ) from exc

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "Không thể parse JSON response từ Ollama."
            ) from exc

    def stream_chat(self, messages: list[Message]):
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "think": False,
            "stream": True
        }

        try:
            with httpx.stream(
                "POST",
                url,
                json=payload,
                timeout=120.0
            ) as response: 
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(
                            f"Không thể parse JSON chunk từ Ollama: {line}"
                        ) from exc

                    content = data.get("message", {}).get("content", "")

                    if content:
                        yield content

        except httpx.ConnectError as exc:
            raise RuntimeError(
                f"Không thể kết nối tới Ollama tại {self.base_url}. "
                "Hãy kiểm tra Ollama server có đang chạy không."
            ) from exc

        except httpx.TimeoutException as exc:
            raise RuntimeError(
                "Ollama streaming bị timeout."
            ) from exc

        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama trả về HTTP {exc.response.status_code}: "
                f"{exc.response.text}"
            ) from exc