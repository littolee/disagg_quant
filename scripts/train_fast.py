"""Single-GPU entrypoint for the authors' QADD trainer with local Tulu parquet."""
import os
import sys
from pathlib import Path

ROOT = Path('/home/litto/work/disagg_quant')
sys.path.insert(0, str(ROOT / 'disaggregated-quantization-main' / 'qad'))
from datasets import load_dataset
import qad_fast_generated as qad

def local_tulu():
    files = sorted(str(p) for p in (ROOT / 'data/allenai__tulu-3-sft-mixture/data').glob('*.parquet'))
    if len(files) != 6:
        raise RuntimeError(f'Expected six Tulu shards, found {len(files)}')
    splits = load_dataset('parquet', data_files=files, split='train').train_test_split(test_size=0.1, seed=42)
    return splits['train'], splits['test']

qad.get_tulu_train_val = local_tulu
qad.main()
