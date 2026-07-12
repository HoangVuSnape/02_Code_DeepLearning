# src/utils/progress.py
"""Progress bar tqdm co fallback — moi truong khong co tqdm van chay."""


def get_pbar(iterable, desc="", disable=False):
    if disable:
        return iterable
    try:
        from tqdm.auto import tqdm
        return tqdm(iterable, desc=desc, leave=False)
    except ImportError:
        return iterable
