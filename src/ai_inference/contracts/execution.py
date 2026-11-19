"""Immutable execution limits, independent of vendor sessions and request state."""

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ShapeRange:
    name: str
    minimum: tuple[int, ...]
    optimum: tuple[int, ...]
    maximum: tuple[int, ...]


@dataclass(frozen=True)
class ExecutionProfile:
    inputs: tuple[ShapeRange, ...]

    def batch_range(self, shapes: Mapping[str, tuple[int, ...]]) -> tuple[int, int] | None:
        if set(shapes) != {item.name for item in self.inputs}:
            return None
        for item in self.inputs:
            shape = shapes[item.name]
            if len(shape) != len(item.minimum) or any(
                not low <= dim <= high
                for dim, low, high in zip(shape[1:], item.minimum[1:], item.maximum[1:])
            ):
                return None
        low = max(item.minimum[0] for item in self.inputs)
        high = min(item.maximum[0] for item in self.inputs)
        return (low, high) if low <= high else None


@dataclass(frozen=True)
class BatchCapabilities:
    profiles: tuple[ExecutionProfile, ...]

    def ranges(self, shapes: Mapping[str, tuple[int, ...]]) -> tuple[tuple[int, int], ...]:
        return tuple(bounds for profile in self.profiles if (bounds := profile.batch_range(shapes)))

    def select_profile(self, shapes: Mapping[str, tuple[int, ...]]) -> int:
        for index, profile in enumerate(self.profiles):
            bounds = profile.batch_range(shapes)
            if bounds and all(bounds[0] <= shape[0] <= bounds[1] for shape in shapes.values()):
                return index
        raise ValueError(f"input shapes do not jointly match any execution profile: {dict(shapes)}")
