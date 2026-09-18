"""
ускоренная версия варианта 0 через numba njit.

алгоритм оставлен как в src/slow_code.py: те же три вложенных цикла, та же копия
матрицы на каждом шаге k. меняется только способ исполнения - циклы компилируются
в машинный код вместо интерпретации.
"""

from __future__ import annotations

import numpy as np
from numba import njit  # type: ignore[import-untyped]

from hw06.optimized import DEFAULT_N, build_distances


@njit(cache=True)
def _floyd_warshall(dist: np.ndarray) -> np.ndarray:
    n = dist.shape[0]
    for k in range(n):
        new_dist = dist.copy()
        for i in range(n):
            for j in range(n):
                candidate = dist[i, k] + dist[k, j]
                if candidate < dist[i, j]:
                    new_dist[i, j] = candidate
                new_dist[i, j] = min(new_dist[i, j], dist[i, j])
        dist = new_dist
    return dist


def accelerated_shortest_paths(n: int = DEFAULT_N) -> list[list[float]]:
    dist = np.array(build_distances(n), dtype=np.float64)
    return _floyd_warshall(dist).tolist()


VARIANTS = {0: accelerated_shortest_paths}


def accelerated_run(variant: int):
    """Запускает ускоренную (Numba) версию алгоритма для данного варианта."""
    if variant not in VARIANTS:
        raise NotImplementedError(f"вариант {variant} не реализован, мой вариант 0")
    return VARIANTS[variant]()
