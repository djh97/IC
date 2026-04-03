from __future__ import annotations

from pathlib import Path
from string import Template

from .config import AppConfig, build_default_config


class PromptLoader:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or build_default_config()

    def load(self, filename: str) -> str:
        path = self.config.paths.prompts_root / filename
        return path.read_text(encoding="utf-8").strip()

    def path(self, filename: str) -> Path:
        return self.config.paths.prompts_root / filename

    def render(self, filename: str, values: dict[str, object]) -> str:
        template = Template(self.load(filename))
        string_values = {key: str(value) for key, value in values.items()}
        return template.safe_substitute(string_values)
