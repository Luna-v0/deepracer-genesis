"""Print the columns of one rollout parquet and the stacks it allows.

Run with ``uv run python -m experiments.perception.visualization.inspect_parquet
<path/rollout_0000.parquet>``.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

K = 4


def stackable_rows(df: pd.DataFrame, k: int = K) -> int:
    """Count rows starting a run of k frames from one car, episode and instant.

    Args:
        df: A rollout shard read from parquet.
        k: Frames per stack.

    Returns:
        The number of valid stack starts.
    """
    if len(df) < k:
        return 0
    env, episode = df["env"].to_numpy(), df["episode"].to_numpy()
    step = df["t"].to_numpy()
    i = np.arange(len(df) - k + 1)
    j = i + k - 1
    return int(((env[i] == env[j]) & (episode[i] == episode[j])
                & (step[j] - step[i] == k - 1)).sum())


def main() -> None:
    """Report one shard's columns, row count and stackable-row count."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = pd.read_parquet(sys.argv[1])
    logger.info("columns: %s", list(df.columns))
    logger.info("rows: %d", len(df))
    logger.info("stackable at k=%d: %d", K, stackable_rows(df))


if __name__ == "__main__":
    main()
