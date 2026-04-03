from __future__ import annotations

from pathlib import Path
import json

from .config import AppConfig, build_default_config
from .public_sources import PublicSourcePlanItem, build_download_plan


class SourceRegistry:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or build_default_config()

    @property
    def default_registry_path(self) -> Path:
        return self.config.paths.configs_root / "source_registry.example.json"

    def resolve_path(self, path: Path | None = None) -> Path:
        return (path or self.default_registry_path).resolve()

    def load(self, path: Path | None = None) -> dict:
        registry_path = self.resolve_path(path)
        return json.loads(registry_path.read_text(encoding="utf-8"))

    def plan(
        self,
        path: Path | None = None,
        *,
        group_ids: set[str] | None = None,
        source_ids: set[str] | None = None,
    ) -> list[PublicSourcePlanItem]:
        registry = self.load(path)
        return build_download_plan(
            registry,
            group_ids=group_ids,
            source_ids=source_ids,
        )
