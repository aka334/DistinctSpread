# k-novel distinct counts: code

Implementation of TNDS and the baselines used in the experiments, for a single
stream and for grouped (multiple) streams.

```
common/code/           hashing, trace loaders, exact ground truth, trace converters
single_stream/
  tnds/code/           TNDS (tnds.py) and its single-stream sweep
  hb/code/             HB, the epoch-bitmap baseline (epoch_bitmap.py)
  theta/code/          Theta-Difference (theta_ring.py)
  hll/code/            HLL-Difference and HLL-Joint (hll_ring.py)
multiple_stream/
  common/code/         grouped-stream input, shared TNDS filter (shared_tnds.py)
  tnds_ht/code/        shared TNDS + exact hash table
  tnds_ssvs/code/      shared TNDS + SSVS counters (ssvs.py)
  bskt_bitmap/code/    bSkt(HB), HB over a shared pool of bitmaps
  bskt_hll/code/       bSkt(HLL-Joint), HLL-Joint over a shared pool of HLL arrays
```

Each `code/` folder has a sweep script that writes estimates to the
neighbouring `result/` folder, and a `test_*.py` file.

## Requirements

Python 3.10+, NumPy 2.x, pandas, SciPy, Numba; a C compiler for
`pcap2bin.c`; Java only for `test_ssvs.py`.

```
pip install -r requirements.txt
```

## Data

The traces are not included. Download them and set `KNOVEL_DATA` to the data
directory (default: `../data` next to this folder):

```
$KNOVEL_DATA/caida/o1.txt ... o62.txt
$KNOVEL_DATA/traces/mawi/202610121400.bin
$KNOVEL_DATA/traces/ecommerce/2019-Oct.bin
```

- **CAIDA**: anonymized Internet traces, https://www.caida.org/catalog/datasets/passive_dataset/
  (access by request). One file per one-minute epoch, one packet per line as
  `src dst` in dotted-quad form.
- **MAWI**: samplepoint-F, 2026-10-12 14:00 JST,
  https://mawi.wide.ad.jp/mawi/samplepoint-F/2026/202610121400.html
  ```
  gcc -O2 -o pcap2bin common/code/pcap2bin.c
  gzip -dc 202610121400.pcap.gz | ./pcap2bin > $KNOVEL_DATA/traces/mawi/202610121400.bin
  ```
- **E-commerce**: REES46 multi-category store, October 2019, https://data.rees46.com
  ```
  python common/code/ecommerce2bin.py 2019-Oct.csv.gz $KNOVEL_DATA/traces/ecommerce/2019-Oct.bin
  ```

Parsed traces and ground truth are cached under `$KNOVEL_DATA/cache`.

## Running


Single stream (CAIDA, each destination its own stream; budgets per flow in KB):

```
python single_stream/tnds/code/sweep_single_stream.py --seed 1
python single_stream/hb/code/sweep_single_stream.py --seed 1
python single_stream/theta/code/sweep_single_stream.py --seed 1
python single_stream/hll/code/sweep_single_stream.py --seed 1
```

Multiple streams (one shared structure; total budgets in MB; `--dataset caida|mawi|ecommerce`):

```
python multiple_stream/tnds_ht/code/sweep.py --dataset caida --seed 1 --counters 1
python multiple_stream/tnds_ssvs/code/sweep.py --dataset caida --seed 1 --per-n \
    --fracs 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --ds 1 2 4
python multiple_stream/bskt_bitmap/code/run_max.py --dataset caida
python multiple_stream/bskt_hll/code/run.py --dataset caida
```

Each run writes `.npz` files with an `est` array of shape (12, F), estimates
for k = 1..12 and every group active at epoch 12 (NaN = unavailable), plus a
`memory.tsv` with the configuration used at each budget. The exact counts are
in `common/code/flow_streams.py` (single stream) and
`multiple_stream/common/code/grouped_streams.py` (grouped streams).

## Tests

```
python single_stream/tnds/code/test_tnds.py
python single_stream/hb/code/test_epoch_bitmap.py
python single_stream/theta/code/test_theta_ring.py
python single_stream/hll/code/test_hll_ring.py
python multiple_stream/common/code/test_shared_tnds.py
python multiple_stream/bskt_bitmap/code/test_bskt_bitmap.py
python multiple_stream/bskt_hll/code/test_bskt_hll.py
python common/code/test_per_flow_cells.py                 # needs CAIDA
python single_stream/hll/code/test_sweep_matches_ring.py  # needs CAIDA
```
