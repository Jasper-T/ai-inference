"""Explicit registries; register plugins before constructing Models."""

from typing import Any, Callable


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._factories: dict[str, Callable] = {}

    def register(self, name: str, factory: Callable | None = None):
        def install(value):
            if not name or name in self._factories:
                raise ValueError(f"Invalid or duplicate {self.kind}: {name!r}")
            self._factories[name] = value
            return value

        return install if factory is None else install(factory)

    def create(self, name: str, config: Any, **kwargs: Any):
        if name not in self._factories:
            raise ValueError(f"Unknown {self.kind} {name!r}; registered: {sorted(self._factories)}")
        return self._factories[name](config, **kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))
