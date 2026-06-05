from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import pandas as pd


DEFAULT_CONDITIONS = ["active", "passive", "constant"]


def _block_picture_count(block: dict) -> int:
    if "pic_samples" in block:
        return len(block["pic_samples"])
    if "images" in block:
        return len(block["images"])
    return 0


def _choose_best_subsequence(longer: list[dict], shorter: list[dict]) -> list[dict]:
    """Choose the contiguous subsequence whose per-block picture counts best match.

    This handles leading practice blocks in EEG cleanly without hard-coding a condition.
    """
    if len(longer) <= len(shorter):
        return list(longer)

    best_slice = list(longer[: len(shorter)])
    best_score = None
    for offset in range(len(longer) - len(shorter) + 1):
        candidate = longer[offset : offset + len(shorter)]
        exact_matches = sum(
            _block_picture_count(a) == _block_picture_count(b)
            for a, b in zip(candidate, shorter)
        )
        abs_diff = sum(
            abs(_block_picture_count(a) - _block_picture_count(b))
            for a, b in zip(candidate, shorter)
        )
        score = (-exact_matches, abs_diff, offset)
        if best_score is None or score < best_score:
            best_score = score
            best_slice = list(candidate)
    return best_slice


def align_condition_blocks(
    eeg_blocks: list[dict],
    et_blocks: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Align EEG and ET blocks before image pairing.

    The main use case is an extra leading EEG practice block (e.g. scrolling/active),
    but this is implemented generically by choosing the best count-matched subsequence.
    """
    eeg_blocks = list(eeg_blocks)
    et_blocks = list(et_blocks)

    if not eeg_blocks or not et_blocks:
        return [], []

    if len(eeg_blocks) == len(et_blocks):
        return eeg_blocks, et_blocks

    if len(eeg_blocks) > len(et_blocks):
        return _choose_best_subsequence(eeg_blocks, et_blocks), et_blocks

    return eeg_blocks, _choose_best_subsequence(et_blocks, eeg_blocks)


def build_sync_pairs(
    eeg_blocks: Iterable[dict],
    et_blocks: Iterable[dict],
    conditions: Iterable[str] | None = None,
) -> pd.DataFrame:
    eeg_by_cond: dict[str, list[dict]] = defaultdict(list)
    for block in eeg_blocks:
        eeg_by_cond[block["block_type"]].append(block)

    et_by_cond: dict[str, list[dict]] = defaultdict(list)
    for block in et_blocks:
        et_by_cond[block["block_type"]].append(block)

    rows: list[dict] = []
    for cond in list(conditions or DEFAULT_CONDITIONS):
        eeg_cond, et_cond = align_condition_blocks(eeg_by_cond[cond], et_by_cond[cond])
        n_blocks = min(len(eeg_cond), len(et_cond))

        for b_idx in range(n_blocks):
            eb = eeg_cond[b_idx]
            tb = et_cond[b_idx]
            n_pics = min(len(eb["pic_samples"]), len(tb["images"]))
            for p in range(n_pics):
                img = tb["images"][p]
                if img["et_off"] is None:
                    continue
                rows.append(
                    dict(
                        block_type=cond,
                        rep=tb["rep"],
                        session=tb["session"],
                        pic_idx=img["pic_idx"],
                        et_on=float(img["et_on"]),
                        et_off=float(img["et_off"]),
                        eeg_pic_s=eb["pic_samples"][p] / 1000.0,
                        eeg_pic_sample=int(eb["pic_samples"][p]),
                    )
                )

    return pd.DataFrame(rows)
