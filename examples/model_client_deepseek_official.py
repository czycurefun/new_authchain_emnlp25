from __future__ import annotations

import ast
import time
from pathlib import Path
from typing import Any

from openai import OpenAI


DEFAULT_API_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
_THIS_FILE = Path(__file__).resolve()
DEFAULT_API_KEY_FILE = str(
    _THIS_FILE.parents[1] / "AuthChain.py" if _THIS_FILE.parent.name == "examples" else _THIS_FILE.with_name("AuthChain.py")
)


def is_placeholder_api_key(value: str) -> bool:
    value = value.strip()
    if not value:
        return True
    placeholder_markers = ("*", "YOUR_", "your_", "PLACEHOLDER", "placeholder", "<", ">")
    return any(marker in value for marker in placeholder_markers)


def load_api_key_from_python_file(file_path: str, variable_name: str = "API_KEY") -> str:
    """Read a simple string constant from a Python file without importing it."""
    if not file_path:
        return ""
    path = Path(file_path)
    if not path.exists():
        return ""

    candidate_names = [variable_name]
    for fallback_name in ("API_KEY", "api_key", "apikey"):
        if fallback_name not in candidate_names:
            candidate_names.append(fallback_name)

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in candidate_names:
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    value = node.value.value.strip()
                    if is_placeholder_api_key(value):
                        return ""
                    return value
    return ""


class RemoteChatClient:
    """DeepSeek official OpenAI SDK adapter.

    Copy this file over model_client.py if you want to use the SDK-style
    implementation from DeepSeek's official docs.
    """

    def __init__(
        self,
        api_base_url: str = DEFAULT_API_BASE_URL,
        api_key: str = "",
        model: str = DEFAULT_MODEL,
        temperature: float = 0.1,
        timeout: int = 60,
        max_retries: int = 3,
        use_env_proxy: bool = False,
        enable_thinking: bool = False,
        system_prompt: str = "",
        request_sleep: float = 0.0,
    ) -> None:
        if use_env_proxy:
            # The OpenAI SDK reads proxy settings from the environment through httpx.
            # This argument is kept for compatibility with the local CLI.
            pass
        self.client = OpenAI(api_key=api_key, base_url=api_base_url, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.enable_thinking = enable_thinking
        self.system_prompt = system_prompt
        self.request_sleep = request_sleep

    def complete(self, prompt: str, **_: Any) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if self.enable_thinking:
            kwargs["reasoning_effort"] = "high"
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
        else:
            kwargs["temperature"] = self.temperature

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                message = response.choices[0].message
                content = getattr(message, "content", "") or ""
                if not content.strip():
                    content = getattr(message, "reasoning_content", "") or ""
                if not content.strip():
                    raise RuntimeError("DeepSeek returned empty content.")
                if self.request_sleep > 0:
                    time.sleep(self.request_sleep)
                return content
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(min(2 * attempt, 8))

        raise RuntimeError(f"DeepSeek request failed after retries: {last_error}") from last_error
