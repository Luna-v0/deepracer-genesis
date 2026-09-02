"""Print the columns of one rollout parquet and the stacks it allows.

A quick look at what a freshly collected file contains, before the dataset is
built from it.

    python -m perception.visualization.inspect_parquet <path/rollout_0000.parquet>
"""

import sys

import pandas as pd

K = 4


def stackable_rows(df, k=K):
    """Rows where the next k frames stay within the same car and episode."""
    env, ep = df["env"].to_numpy(), df["episode"].to_numpy()
    return [i for i in range(len(df) - k + 1)
            if env[i] == env[i + k - 1] and ep[i] == ep[i + k - 1]]


def main():
    df = pd.read_parquet(sys.argv[1])
    print("columns:", list(df.columns))
    print(f"rows: {len(df)}")
    print(f"stackable at k={K}: {len(stackable_rows(df))}")


if __name__ == "__main__":
    main()
