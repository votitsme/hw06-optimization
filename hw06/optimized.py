"""
оптимизированная версия варианта 0 (кратчайшие пути между всеми парами).

три итерации оптимизации из hw06/REPORT.md оставлены отдельными функциями,
чтобы замеры можно было повторить. optimized_run использует последнюю.
"""

from __future__ import annotations

import random

import numpy as np

INF = float("inf")
DEFAULT_N = 200


def build_distances(n: int = DEFAULT_N) -> list[list[float]]:
    # порядок вызовов random тот же, что в src/slow_code.py, иначе граф выйдет другой
    random.seed(42)
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0.0
    for _ in range(n * 3):
        u, v = random.randint(0, n - 1), random.randint(0, n - 1)
        w = random.uniform(1.0, 100.0)
        if w < dist[u][v]:
            dist[u][v] = w
    return dist


def shortest_paths_inplace(n: int = DEFAULT_N) -> list[list[float]]:
    """итерация 1: обновляем матрицу на месте, без копий и без лишнего min. O(n^3)"""
    dist = build_distances(n)
    for k in range(n):
        row_k = dist[k]
        for row_i in dist:
            d_ik = row_i[k]
            for j in range(n):
                candidate = d_ik + row_k[j]
                if candidate < row_i[j]:
                    row_i[j] = candidate
    return dist


def shortest_paths_skip_unreachable(n: int = DEFAULT_N) -> list[list[float]]:
    """итерация 2: если k недостижим из i, вся внутренняя строка бесполезна"""
    dist = build_distances(n)
    for k in range(n):
        row_k = dist[k]
        for row_i in dist:
            d_ik = row_i[k]
            if d_ik == INF:
                continue
            for j in range(n):
                candidate = d_ik + row_k[j]
                if candidate < row_i[j]:
                    row_i[j] = candidate
    return dist


def shortest_paths_numpy(n: int = DEFAULT_N) -> list[list[float]]:
    """итерация 3: внутренние два цикла заменены на один np.minimum по всей матрице"""
    dist = np.array(build_distances(n), dtype=np.float64)
    for k in range(n):
        np.minimum(dist, dist[:, k, None] + dist[k], out=dist)
    return dist.tolist()


VARIANTS = {0: shortest_paths_numpy}


def optimized_run(variant: int):
    """Запускает оптимизированную версию алгоритма для данного варианта."""
    if variant not in VARIANTS:
        raise NotImplementedError(f"вариант {variant} не реализован, мой вариант 0")
    return VARIANTS[variant]()
