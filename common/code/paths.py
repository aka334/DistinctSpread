"""Data locations. Set KNOVEL_DATA to the data directory (default: ../data
next to this repository's code/ folder). Expected layout:

    $KNOVEL_DATA/caida/o1.txt ... o62.txt
    $KNOVEL_DATA/traces/mawi/202610121400.bin
    $KNOVEL_DATA/traces/ecommerce/2019-Oct.bin
"""

import os
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("KNOVEL_DATA", CODE.parent / "data")).resolve()
CAIDA = DATA / "caida"
TRACES = DATA / "traces"
CACHE = DATA / "cache"
