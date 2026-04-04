from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class ModelConfig:
    provider: str = "huggingface"
    generator_model: str = "Qwen/Qwen3-8B"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str | None = None
    endpoint_url: str | None = None
    temperature: float = 0.1
    max_tokens: int = 2048


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    chunk_size: int = 1200
    chunk_overlap: int = 150
    top_k: int = 5
    similarity_metric: str = "cosine"
    retrieval_mode: str = "hybrid"
    rrf_k: int = 60


@dataclass(frozen=True, slots=True)
class PathConfig:
    project_root: Path = Path(".")
    artifact_root: Path = Path("artifacts")
    source_data_root: Path = Path("data")
    configs_root: Path = Path("configs")
    prompts_root: Path = Path("prompts")
    docs_root: Path = Path("docs")
    scripts_root: Path = Path("scripts")


@dataclass(frozen=True, slots=True)
class AppConfig:
    study_id: str = "STUDY-0001"
    site_id: str = "SITE-1"
    models: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    paths: PathConfig = field(default_factory=PathConfig)


def build_default_config(project_root: str | Path | None = None) -> AppConfig:
    load_dotenv()
    root = Path(project_root or ".").resolve()
    model_endpoint = os.getenv("HF_INFERENCE_ENDPOINT") or os.getenv("HF_ENDPOINT_URL")
    model_defaults = ModelConfig()
    generator_model = os.getenv("HF_MODEL_ID", model_defaults.generator_model).strip() or model_defaults.generator_model
    embedding_model = (
        os.getenv("IC_EMBEDDING_MODEL", model_defaults.embedding_model).strip() or model_defaults.embedding_model
    )
    reranker_model = os.getenv("IC_RERANKER_MODEL", "").strip() or None

    return AppConfig(
        models=ModelConfig(
            generator_model=generator_model,
            embedding_model=embedding_model,
            reranker_model=reranker_model,
            endpoint_url=model_endpoint,
        ),
        paths=PathConfig(
            project_root=root,
            artifact_root=root / "artifacts",
            source_data_root=root / "data",
            configs_root=root / "configs",
            prompts_root=root / "prompts",
            docs_root=root / "docs",
            scripts_root=root / "scripts",
        ),
    )
