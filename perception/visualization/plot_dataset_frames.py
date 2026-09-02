"""Contact sheet of the camera frames collected for one track.

    python -m perception.visualization.plot_dataset_frames <dataset folder>
"""

import glob
import io
import sys
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
ROWS, COLUMNS = 3, 4


def main():
    folder = sys.argv[1]
    files = sorted(glob.glob(str(REPO_ROOT / "data" / folder / "*.parquet")))
    table = pq.read_table(files[0])
    step = table.num_rows // (ROWS * COLUMNS)   # spread the picks over the file

    sheet = Image.new("RGB", (160 * COLUMNS, 120 * ROWS))
    for n in range(ROWS * COLUMNS):
        img = Image.open(io.BytesIO(table["image"][n * step].as_py()))
        sheet.paste(img, (160 * (n % COLUMNS), 120 * (n // COLUMNS)))

    out = REPO_ROOT / "runs" / "figures" / f"dataset_frames_{folder}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print("open:", out)


if __name__ == "__main__":
    main()
