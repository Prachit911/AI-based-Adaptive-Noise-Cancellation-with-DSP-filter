"""Unit tests for the manifest generation and leakage prevention."""
from __future__ import annotations

import pandas as pd
import pytest

from noise_suppression.constants import MANIFEST_COLUMNS
from noise_suppression.make_manifest import check_leakage


def test_check_leakage_clean():
    # Construct a valid, non-leaking dataframe
    data = {
        "mixture_id": ["m1", "m2", "m3"],
        "split": ["train", "val", "test"],
        "clean_file": ["c1.wav", "c2.wav", "c3.wav"],
        "noise_file": ["n1.wav", "n2.wav", "n3.wav"],
        "noise_file_2": ["", "", ""],
        "noise_category": ["rain", "engine", "wind"],
        "noise_category_2": ["", "", ""],
        "esc50_fold": [1, 4, 5],
        "snr_db": [0.0, 5.0, 10.0],
        "seed": [101, 102, 103],
        "speaker_id": ["spk_A", "spk_B", "spk_C"],
    }
    df = pd.DataFrame(data)
    assert check_leakage(df) is True


def test_check_leakage_detects_speaker_overlap():
    data = {
        "mixture_id": ["m1", "m2"],
        "split": ["train", "test"],
        "clean_file": ["c1.wav", "c2.wav"],
        "noise_file": ["n1.wav", "n2.wav"],
        "noise_file_2": ["", ""],
        "noise_category": ["rain", "wind"],
        "noise_category_2": ["", ""],
        "esc50_fold": [1, 5],
        "snr_db": [0.0, 5.0],
        "seed": [101, 102],
        "speaker_id": ["spk_LEAK", "spk_LEAK"],  # Same speaker in train and test!
    }
    df = pd.DataFrame(data)
    assert check_leakage(df) is False


def test_check_leakage_detects_fold_overlap():
    data = {
        "mixture_id": ["m1", "m2"],
        "split": ["train", "test"],
        "clean_file": ["c1.wav", "c2.wav"],
        "noise_file": ["n1.wav", "n2.wav"],
        "noise_file_2": ["", ""],
        "noise_category": ["rain", "wind"],
        "noise_category_2": ["", ""],
        "esc50_fold": [5, 5],  # Fold 5 in both train and test!
        "snr_db": [0.0, 5.0],
        "seed": [101, 102],
        "speaker_id": ["spk_A", "spk_B"],
    }
    df = pd.DataFrame(data)
    assert check_leakage(df) is False
