"""Convert the REES46 2019-Oct.csv.gz event log to 12-byte records
<ts_sec, user_id, product_id> (little-endian uint32), all event types.

    python ecommerce2bin.py 2019-Oct.csv.gz $KNOVEL_DATA/traces/ecommerce/2019-Oct.bin
"""

import sys

import numpy as np
import pandas as pd


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    n = 0
    with open(dst, "wb") as out:
        for df in pd.read_csv(src, usecols=["event_time", "product_id", "user_id"], chunksize=5_000_000):
            ts = (pd.to_datetime(df.event_time.str[:19], format="%Y-%m-%d %H:%M:%S").astype("int64") // 10**9).to_numpy()
            u, p = df.user_id.to_numpy(np.int64), df.product_id.to_numpy(np.int64)
            assert u.max() < 2**32 and p.max() < 2**32
            np.stack([ts, u, p], axis=1).astype("<u4").tofile(out)
            n += len(df)
    print(f"{n:,} events")


if __name__ == "__main__":
    main()
