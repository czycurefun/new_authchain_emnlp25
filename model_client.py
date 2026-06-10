from __future__ import annotations

import ast
import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


DEFAULT_API_BASE_URL = "https://antchat.alipay.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_API_KEY_FILE = str(Path(__file__).resolve().with_name("AuthChain.py"))


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


def extract_model_text(response_json: Dict[str, Any]) -> tuple[str, str]:
    """Prefer standard message.content, then fall back to reasoning_content."""
    choices = response_json.get("choices") if isinstance(response_json, dict) else None
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, str) and content.strip():
        return content, "content"
    reasoning_content = message.get("reasoning_content", "") if isinstance(message, dict) else ""
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return reasoning_content, "reasoning_content"
    return "", "content"


def build_response_meta(response_json: Dict[str, Any], status_code: Optional[int]) -> Dict[str, Any]:
    choices = response_json.get("choices") if isinstance(response_json, dict) else None
    first_choice = choices[0] if isinstance(choices, list) and choices else {}
    message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    reasoning_content = message.get("reasoning_content", "") if isinstance(message, dict) else ""
    return {
        "status_code": status_code,
        "choices_count": len(choices) if isinstance(choices, list) else None,
        "finish_reason": first_choice.get("finish_reason") if isinstance(first_choice, dict) else None,
        "message_keys": sorted(message.keys()) if isinstance(message, dict) else [],
        "content_chars": len(content) if isinstance(content, str) else None,
        "reasoning_content_chars": len(reasoning_content) if isinstance(reasoning_content, str) else None,
        "usage": response_json.get("usage") if isinstance(response_json, dict) else None,
        "error": response_json.get("error") if isinstance(response_json, dict) else None,
    }


def call_chat_completion(
    api_base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.1,
    timeout: int = 60,
    max_retries: int = 3,
    use_env_proxy: bool = False,
    enable_thinking: bool = False,
) -> Dict[str, Any]:
    """Call an AntChat/OpenAI-compatible Chat Completions endpoint."""
    if not api_key:
        raise ValueError("Missing API key.")

    endpoint = api_base_url.rstrip("/") + "/v1/chat/completions"
    is_deepseek_official = "api.deepseek.com" in api_base_url
    request_body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
    }
    if is_deepseek_official:
        if enable_thinking:
            request_body["thinking"] = {"type": "enabled"}
            request_body["reasoning_effort"] = "high"
    else:
        request_body["enable_thinking"] = enable_thinking

    session = requests.Session()
    session.trust_env = use_env_proxy
    last_error = None
    for attempt in range(1, max_retries + 1):
        trace_id = str(uuid.uuid4())
        try:
            response = session.post(
                endpoint,
                json=request_body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "SOFA-TraceId": trace_id,
                    "SOFA-RpcId": "0.1",
                },
                timeout=timeout,
            )
            response.raise_for_status()
            response_json = response.json()
            content, content_source = extract_model_text(response_json)
            response_meta = build_response_meta(response_json, response.status_code)
            response_meta["used_message_field"] = content_source
            if not content.strip():
                raise RuntimeError(f"Model returned empty content: {json.dumps(response_meta, ensure_ascii=False)}")
            return {
                "ok": True,
                "trace_id": trace_id,
                "content": content,
                "content_source": content_source,
                "response_meta": response_meta,
                "response_json": response_json,
            }
        except (requests.RequestException, KeyError, json.JSONDecodeError, RuntimeError) as exc:
            status_code = None
            response_text = ""
            if isinstance(exc, requests.HTTPError) and exc.response is not None:
                status_code = exc.response.status_code
                response_text = exc.response.text[:1000]
            last_error = f"{type(exc).__name__}: {exc}"
            if response_text:
                last_error += f"; response: {response_text}"
            if attempt < max_retries:
                if status_code == 429 or "LIMIT_EXCEEDED" in response_text or "额度超限" in response_text:
                    time.sleep(min(20 * attempt, 90))
                else:
                    time.sleep(min(2 * attempt, 8))

    return {
        "ok": False,
        "trace_id": "",
        "content": "",
        "content_source": "content",
        "response_meta": {},
        "response_json": None,
        "error": last_error or "unknown request error",
    }


class RemoteChatClient:
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
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self.use_env_proxy = use_env_proxy
        self.enable_thinking = enable_thinking
        self.system_prompt = system_prompt
        self.request_sleep = request_sleep

    def complete(self, prompt: str, **_: Any) -> str:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        result = call_chat_completion(
            api_base_url=self.api_base_url,
            api_key=self.api_key,
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            timeout=self.timeout,
            max_retries=self.max_retries,
            use_env_proxy=self.use_env_proxy,
            enable_thinking=self.enable_thinking,
        )
        if not result["ok"]:
            raise RuntimeError(result.get("error") or "model call failed")
        content = result["content"]
        if self.request_sleep > 0:
            time.sleep(self.request_sleep)
        return content
