"""Content-addressed caches.

Two caches, one mechanism. Both key on a SHA-256 of everything that could change the
answer, so a hit is always safe and a miss is always necessary.

* **OCR cache** (in :mod:`throughline.ingest.ocr`) keys on the source bytes.
* **Prompt cache** (here) keys on the assembled prompt plus the decoding config plus
  the backend name.

The prompt cache is what makes iteration on the *rest* of the system cheap. Changing
the merge policy, the early-exit threshold, or the attribution ladder does not change
any prompt, so a re-run costs nothing in model calls. It also deduplicates within a
run: overlapping page groups in near-identical documents - the same invoice template
filed 400 times - hit the same entries.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from throughline.models.base import GenerationConfig, GenerationResult
from throughline.prompting.templates import PromptBundle

LOGGER = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Hit/miss accounting, reported at the end of a run."""

    hits: int = 0
    misses: int = 0
    stores: int = 0
    bytes_written: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "stores": self.stores,
            "hit_rate": round(self.hit_rate, 4),
            "bytes_written": self.bytes_written,
        }


def prompt_cache_key(
    prompt: PromptBundle, config: GenerationConfig, backend_name: str
) -> str:
    """SHA-256 over prompt text, image paths, decoding config and backend."""
    digest = hashlib.sha256()
    digest.update(backend_name.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(prompt.cache_key_material().encode("utf-8"))
    digest.update(b"\x00")
    digest.update(json.dumps(config.to_dict(), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


@dataclass
class PromptCache:
    """On-disk cache of backend responses, sharded by key prefix.

    Sharding matters at corpus scale: a flat directory with 100k entries is slow to
    list on most filesystems, and every debugging session lists it.
    """

    cache_dir: str | Path = ".cache/prompts"
    enabled: bool = True
    ttl_seconds: float | None = None
    stats: CacheStats = field(default_factory=CacheStats)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def _path_for(self, key: str) -> Path:
        return Path(self.cache_dir) / key[:2] / f"{key}.json"

    def get(self, key: str) -> GenerationResult | None:
        if not self.enabled:
            return None

        path = self._path_for(key)
        if not path.exists():
            with self._lock:
                self.stats.misses += 1
            return None

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Corrupt prompt-cache entry, ignoring: %s", path)
            with self._lock:
                self.stats.misses += 1
            return None

        if self.ttl_seconds is not None:
            age = time.time() - float(payload.get("stored_at", 0))
            if age > self.ttl_seconds:
                with self._lock:
                    self.stats.misses += 1
                return None

        with self._lock:
            self.stats.hits += 1

        return GenerationResult(
            text=payload["text"],
            prompt_tokens=int(payload.get("prompt_tokens", 0)),
            completion_tokens=int(payload.get("completion_tokens", 0)),
            latency_seconds=0.0,
            cached=True,
            backend=payload.get("backend", ""),
            metadata=payload.get("metadata", {}),
        )

    def put(self, key: str, result: GenerationResult) -> None:
        if not self.enabled:
            return

        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps(
            {
                "text": result.text,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "backend": result.backend,
                "metadata": result.metadata,
                "stored_at": time.time(),
            },
            indent=2,
        )
        # Write-then-rename so a crash mid-write cannot leave a torn entry.
        temporary = path.with_suffix(".tmp")
        temporary.write_text(body, encoding="utf-8")
        temporary.replace(path)

        with self._lock:
            self.stats.stores += 1
            self.stats.bytes_written += len(body)

    def clear(self) -> int:
        """Delete every entry. Returns how many were removed."""
        root = Path(self.cache_dir)
        if not root.exists():
            return 0
        removed = 0
        for path in root.rglob("*.json"):
            path.unlink()
            removed += 1
        return removed

    def size(self) -> tuple[int, int]:
        """(entry count, total bytes) currently on disk."""
        root = Path(self.cache_dir)
        if not root.exists():
            return (0, 0)
        entries = list(root.rglob("*.json"))
        return (len(entries), sum(path.stat().st_size for path in entries))


@dataclass
class CachedBackend:
    """Wrap any :class:`~throughline.models.base.VLMBackend` with the prompt cache."""

    backend: Any
    cache: PromptCache = field(default_factory=PromptCache)

    @property
    def name(self) -> str:
        return f"cached:{self.backend.name}"

    def generate(
        self, prompt: PromptBundle, config: GenerationConfig | None = None
    ) -> GenerationResult:
        config = config or GenerationConfig()
        key = prompt_cache_key(prompt, config, self.backend.name)

        hit = self.cache.get(key)
        if hit is not None:
            LOGGER.debug("Prompt cache hit for group %s", prompt.group_index)
            return hit

        result = self.backend.generate(prompt, config)
        self.cache.put(key, result)
        return result
