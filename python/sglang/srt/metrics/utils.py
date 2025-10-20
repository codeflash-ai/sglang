# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for Prometheus Metrics."""
import math
from typing import List


def two_sides_exponential_buckets(
    middle: float, base: float, count: int
) -> List[float]:
    # Pre-size the list for speed and use a set directly to deduplicate,
    # then avoid repeated sorting - sort only at the end.
    buckets_set = set()
    half_count = math.ceil(count / 2)
    distance = 1
    buckets_set.add(middle)
    for _ in range(half_count):
        distance *= base
        buckets_set.add(middle + distance)
        buckets_set.add(max(0, middle - distance))
    # Sorting is required for return format
    return sorted(buckets_set)


def generate_buckets(
    buckets_rule: List[str], default_buckets: List[float]
) -> List[float]:
    if not buckets_rule:
        buckets_rule = ["default"]

    assert len(buckets_rule) > 0
    rule = buckets_rule[0]
    if rule == "tse":
        middle_str, base_str, count_str = buckets_rule[1:]
        base = float(base_str)
        assert base > 1.0, "Base must be greater than 1.0"
        middle = float(middle_str)
        count = int(count_str)
        return two_sides_exponential_buckets(middle, base, count)
    if rule == "default":
        # Avoid repeated conversion and copying for default buckets
        return sorted(set(default_buckets))
    assert rule == "custom"
    # Use generator (map) for faster float conversion and avoid extra list allocation
    return sorted(set(map(float, buckets_rule[1:])))


def exponential_buckets(start: float, width: float, length: int) -> List[float]:
    buckets = []
    for i in range(length):
        buckets.append(start * (width**i))
    return buckets
