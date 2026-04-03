from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import requests


@dataclass(frozen=True, slots=True)
class HuggingFaceRuntimeConfig:
    token: str
    model_id: str
    endpoint_url: str | None
    enable_thinking: bool
    request_timeout: float | None


def load_hf_runtime_config() -> HuggingFaceRuntimeConfig:
    load_dotenv()

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Copy .env.example to .env and add your token.")

    model_id = os.environ.get("HF_MODEL_ID", "Qwen/Qwen3-8B").strip() or "Qwen/Qwen3-8B"
    endpoint_url = (
        os.environ.get("HF_INFERENCE_ENDPOINT", "").strip()
        or os.environ.get("HF_ENDPOINT_URL", "").strip()
        or None
    )
    enable_thinking = os.environ.get("HF_ENABLE_THINKING", "false").strip().lower() == "true"
    timeout_raw = os.environ.get("HF_REQUEST_TIMEOUT", "").strip()
    request_timeout = None
    if timeout_raw:
        try:
            request_timeout = float(timeout_raw)
        except ValueError as exc:
            raise RuntimeError("HF_REQUEST_TIMEOUT must be numeric if set.") from exc
    else:
        request_timeout = 120.0

    return HuggingFaceRuntimeConfig(
        token=token,
        model_id=model_id,
        endpoint_url=endpoint_url,
        enable_thinking=enable_thinking,
        request_timeout=request_timeout,
    )


def build_hf_client(runtime_config: HuggingFaceRuntimeConfig | None = None) -> InferenceClient:
    runtime_config = runtime_config or load_hf_runtime_config()

    if runtime_config.endpoint_url:
        return InferenceClient(
            base_url=runtime_config.endpoint_url,
            api_key=runtime_config.token,
            timeout=runtime_config.request_timeout,
        )

    return InferenceClient(
        model=runtime_config.model_id,
        api_key=runtime_config.token,
        timeout=runtime_config.request_timeout,
    )


def chat_json(
    messages: list[dict[str, str]],
    schema_name: str,
    schema: dict[str, Any],
    *,
    runtime_config: HuggingFaceRuntimeConfig | None = None,
    temperature: float = 0.1,
    max_tokens: int = 800,
) -> dict[str, Any]:
    runtime_config = runtime_config or load_hf_runtime_config()
    client = build_hf_client(runtime_config)

    request_kwargs: dict[str, object] = {}
    if runtime_config.enable_thinking:
        request_kwargs["extra_body"] = {"enable_thinking": True}

    last_exception: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = client.chat_completion(
                messages=messages,
                model=runtime_config.model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema,
                    },
                },
                **request_kwargs,
            )
            break
        except requests.exceptions.RequestException as exc:
            last_exception = exc
            if attempt == 3:
                raise RuntimeError(
                    f"Hugging Face endpoint request failed after {attempt} attempts."
                ) from exc
            time.sleep(2 * attempt)
    else:  # pragma: no cover - defensive fallback
        raise RuntimeError("Hugging Face endpoint request failed before any response was received.") from last_exception

    content = response.choices[0].message.content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model did not return valid JSON: {content}") from exc

    return parsed
