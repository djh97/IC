from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import requests

from informed_consent.hf_client import HuggingFaceRuntimeConfig, chat_json


class HuggingFaceClientDiagnosticsTests(unittest.TestCase):
    def test_chat_json_surfaces_connection_error_details(self) -> None:
        runtime_config = HuggingFaceRuntimeConfig(
            token="token",
            model_id="Qwen/Qwen3-8B",
            endpoint_url="https://endpoint.example.test",
            enable_thinking=False,
            request_timeout=15.0,
        )
        client = Mock()
        client.chat_completion.side_effect = requests.exceptions.ConnectionError("connection refused")

        with patch("informed_consent.hf_client.build_hf_client", return_value=client), patch(
            "informed_consent.hf_client.time.sleep", return_value=None
        ):
            with self.assertRaises(RuntimeError) as ctx:
                chat_json(
                    messages=[{"role": "user", "content": "hello"}],
                    schema_name="demo",
                    schema={"type": "object"},
                    runtime_config=runtime_config,
                )

        message = str(ctx.exception)
        self.assertIn("endpoint=https://endpoint.example.test", message)
        self.assertIn("model=Qwen/Qwen3-8B", message)
        self.assertIn("ConnectionError", message)
        self.assertIn("connection refused", message)

    def test_chat_json_surfaces_http_status_and_body_preview(self) -> None:
        runtime_config = HuggingFaceRuntimeConfig(
            token="token",
            model_id="Qwen/Qwen3-8B",
            endpoint_url="https://endpoint.example.test",
            enable_thinking=False,
            request_timeout=15.0,
        )
        response = requests.Response()
        response.status_code = 403
        response.url = "https://endpoint.example.test/v1/chat/completions"
        response._content = b'{"error":"forbidden"}'
        client = Mock()
        client.chat_completion.side_effect = requests.exceptions.HTTPError("forbidden", response=response)

        with patch("informed_consent.hf_client.build_hf_client", return_value=client), patch(
            "informed_consent.hf_client.time.sleep", return_value=None
        ):
            with self.assertRaises(RuntimeError) as ctx:
                chat_json(
                    messages=[{"role": "user", "content": "hello"}],
                    schema_name="demo",
                    schema={"type": "object"},
                    runtime_config=runtime_config,
                )

        message = str(ctx.exception)
        self.assertIn("HTTPError", message)
        self.assertIn("status=403", message)
        self.assertIn('body={"error":"forbidden"}', message)


if __name__ == "__main__":
    unittest.main()
