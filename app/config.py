"""Runtime settings. Secrets come from the environment / .env only."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    provider: str
    openai_api_key: str | None
    openai_base_url: str | None
    chat_model: str
    embed_model: str
    context_window: int
    data_dir: Path

    @property
    def db_path(self) -> str:
        return str(self.data_dir / "rag.db")

    @property
    def threads_db_path(self) -> str:
        return str(self.data_dir / "threads.db")

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def rag_settings_path(self) -> Path:
        return self.data_dir / "rag_settings.json"


@dataclass
class RagSettings:
    """Defaults copied from Jan 0.8.6 @janhq/rag-extension."""
    enabled: bool = True
    parse_mode: str = "auto"  # auto | inline | embeddings | prompt
    auto_inline_context_ratio: float = 0.75
    max_file_size_mb: int = 100
    # Jan uses 3; multi-part questions over spreadsheets missed the second fact at 3 and got it at 5.
    retrieval_limit: int = 5
    # Jan uses 0.3 for its local MiniLM model. With text-embedding-3-small, relevant chunks scored
    # 0.25-0.65 and off-topic queries <= 0.13 (scripts/eval_retrieval.py), so 0.2 keeps both apart.
    retrieval_threshold: float = 0.2
    chunk_size_chars: int = 512
    overlap_chars: int = 64
    search_mode: str = "auto"  # auto | ann | linear

    def update(self, data: dict) -> "RagSettings":
        valid = {f.name: f.type for f in fields(self)}
        merged = asdict(self)
        merged.update({k: v for k, v in data.items() if k in valid})
        rs = RagSettings(**merged)
        rs.validate()
        return rs

    def validate(self) -> None:
        if self.parse_mode not in {"auto", "inline", "embeddings", "prompt"}:
            raise ValueError("parse_mode must be auto, inline, embeddings or prompt")
        if self.search_mode not in {"auto", "ann", "linear"}:
            raise ValueError("search_mode must be auto, ann or linear")
        if not 0.05 <= float(self.auto_inline_context_ratio) <= 1:
            raise ValueError("auto_inline_context_ratio must be between 0.05 and 1")
        if not 1 <= int(self.max_file_size_mb) <= 200:
            raise ValueError("max_file_size_mb must be between 1 and 200")
        if not 1 <= int(self.retrieval_limit) <= 20:
            raise ValueError("retrieval_limit must be between 1 and 20")
        if not 0 <= float(self.retrieval_threshold) <= 1:
            raise ValueError("retrieval_threshold must be between 0 and 1")
        if int(self.chunk_size_chars) < 64:
            raise ValueError("chunk_size_chars must be at least 64")
        if not 0 <= int(self.overlap_chars) < int(self.chunk_size_chars):
            raise ValueError("overlap_chars must be >= 0 and smaller than chunk_size_chars")


def load_rag_settings(path: Path) -> RagSettings:
    if path.exists():
        try:
            return RagSettings().update(json.loads(path.read_text("utf-8")))
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    return RagSettings()


def save_rag_settings(path: Path, rs: RagSettings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(rs), indent=2), "utf-8")


@lru_cache
def get_settings() -> Settings:
    data_dir = Path(os.getenv("DATA_DIR") or ROOT / "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        provider=os.getenv("PROVIDER", "openai").lower(),
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        chat_model=os.getenv("CHAT_MODEL", "gpt-4.1-mini"),
        embed_model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        context_window=int(os.getenv("CONTEXT_WINDOW", "128000")),
        data_dir=data_dir,
    )
