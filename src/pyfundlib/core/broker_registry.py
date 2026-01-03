# src/pyfundlib/core/broker_registry.py
from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, Optional

from pyfundlib.utils.logger import get_logger
from .broker import Broker

logger = get_logger(__name__)

# Thread-safe global registries
_BROKER_REGISTRY: dict[str, type[Broker]] = {}
_FETCHER_REGISTRY: dict[str, Callable] = {}
_registry_lock = threading.RLock()


def register_broker(
    name: Optional[str] = None, *, override: bool = False
) -> Callable[[type[Broker]], type[Broker]]:
    """Decorator to register a broker class."""

    def decorator(cls: type[Broker]) -> type[Broker]:
        if not issubclass(cls, Broker):
            raise TypeError(f"Class {cls.__name__} must inherit from Broker")

        broker_name = name or cls.__name__.replace("Broker", "").lower()

        with _registry_lock:
            if broker_name in _BROKER_REGISTRY:
                if not override:
                    logger.warning("broker_already_registered", name=broker_name)
                    raise ValueError(
                        f"Broker '{broker_name}' already registered. Use override=True to replace."
                    )
                else:
                    logger.info("overriding_broker", name=broker_name)

            _BROKER_REGISTRY[broker_name] = cls
            setattr(cls, "_registered_name", broker_name)

        logger.info("broker_registered", name=broker_name)
        return cls

    return decorator


def get_broker(name: str, **credentials: Any) -> Broker:
    """Factory function to instantiate a broker by name."""
    broker_name = name.lower().strip()

    with _registry_lock:
        broker_class = _BROKER_REGISTRY.get(broker_name)

    if not broker_class:
        available = sorted(_BROKER_REGISTRY.keys())
        logger.error("broker_not_found", name=broker_name, available=available)
        raise ValueError(f"Broker '{broker_name}' not found!")

    try:
        instance = broker_class(**credentials)
        return instance
    except Exception as e:
        logger.error("broker_init_failed", name=broker_name, error=str(e))
        raise RuntimeError(f"Failed to initialize broker '{broker_name}': {e}") from e


def list_available_brokers() -> list[str]:
    with _registry_lock:
        return sorted(_BROKER_REGISTRY.keys())


def register_data_fetcher(name: str, fetcher_func: Callable) -> None:
    with _registry_lock:
        _FETCHER_REGISTRY[name.lower()] = fetcher_func
    logger.info("data_fetcher_registered", name=name)


def get_data_fetcher(name: str) -> Callable:
    with _registry_lock:
        fetcher = _FETCHER_REGISTRY.get(name.lower())
    if not fetcher:
        logger.error("fetcher_not_found", name=name)
        raise ValueError(f"Data fetcher '{name}' not found!")
    return fetcher


def clear_registry() -> None:
    with _registry_lock:
        _BROKER_REGISTRY.clear()


def _auto_discover_brokers() -> None:
    import importlib
    from pathlib import Path

    brokers_path = Path(__file__).parent.parent / "brokers"
    if not brokers_path.exists():
        return

    for file in brokers_path.glob("*.py"):
        if file.name.startswith("_") or file.name == "base.py":
            continue

        module_name = f"pyfundlib.brokers.{file.stem}"
        try:
            importlib.import_module(module_name)
        except Exception as e:
            logger.warning("broker_auto_import_failed", module=module_name, error=str(e))


# Auto-discover on import
_auto_discover_brokers()

__all__ = [
    "Broker",
    "clear_registry",
    "get_broker",
    "list_available_brokers",
    "register_broker",
    "register_data_fetcher",
    "get_data_fetcher",
]
