# src/pyfundlib/utils/cache.py
from __future__ import annotations

import hashlib
import pickle
import zlib
from pathlib import Path
from typing import Callable, Any, Optional
from functools import wraps
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def stable_hash(obj: Any) -> str:
    """Deterministic hash for any picklable object"""
    pickled = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    compressed = zlib.compress(pickled)
    return hashlib.sha256(compressed).hexdigest()

class CachedFunction:
    """
    Production-ready, safe, configurable cache decorator.
    """
    def __init__(
        self,
        dir_name: str,
        *,
        expire_days: Optional[int] = 30,
        max_size_mb: Optional[int] = 5000,  # 5GB default
        compress: bool = True,
    ):
        self.cache_dir = Path("./cache") / dir_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.expire_days = expire_days
        self.max_size_mb = max_size_mb
        self.compress = compress

        # Optional: enforce size limit
        if max_size_mb:
            self._enforce_size_limit()

    def _get_cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl{'z' if self.compress else ''}"

    def _enforce_size_limit(self):
        """Remove oldest files if over size limit"""
        files = sorted(self.cache_dir.iterdir(), key=lambda p: p.stat().st_mtime)
        total_size = sum(f.stat().st_size for f in files) / (1024 ** 2)  # MB

        while total_size > self.max_size_mb and len(files) > 1:
            oldest = files.pop(0)
            size_removed = oldest.stat().st_size / (1024 ** 2)
            oldest.unlink(missing_ok=True)
            total_size -= size_removed
            logger.debug(f"Cache pruned: {oldest.name}")

    def _is_expired(self, filepath: Path) -> bool:
        if self.expire_days is None:
            return False
        age = datetime.now() - datetime.fromtimestamp(filepath.stat().st_mtime)
        return age > timedelta(days=self.expire_days)

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create stable, unique key
            key_input = {
                "func_name": func.__name__,
                "args": args,
                "kwargs": kwargs,
            }
            cache_key = stable_hash(key_input)
            cache_file = self._get_cache_path(cache_key)

            # Cache hit
            if cache_file.exists() and not self._is_expired(cache_file):
                try:
                    with open(cache_file, "rb") as f:
                        data = f.read()
                        if self.compress:
                            data = zlib.decompress(data)
                        result = pickle.loads(data)
                    logger.debug(f"Cache HIT: {func.__name__} → {cache_file.name}")
                    return result
                except Exception as e:
                    logger.warning(f"Cache corrupted, recomputing: {e}")

            # Cache miss → compute
            logger.debug(f"Cache MISS: {func.__name__}")
            result = func(*args, **kwargs)

            # Save to cache
            try:
                data = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
                if self.compress:
                    data = zlib.compress(data)
                with open(cache_file, "wb") as f:
                    f.write(data)
                if self.max_size_mb:
                    self._enforce_size_limit()
            except Exception as e:
                logger.error(f"Failed to cache result: {e}")

            return result

        wrapper.cache_dir = self.cache_dir
        wrapper.clear_cache = lambda: [f.unlink() for f in self.cache_dir.glob("*.pkl*")]
        return wrapper

# Convenience decorator
def cached_function(
    dir_name: str,
    expire_days: int = 30,
    max_size_mb: int = 5000,
):
    return CachedFunction(dir_name, expire_days=expire_days, max_size_mb=max_size_mb)