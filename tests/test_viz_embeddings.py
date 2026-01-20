from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

EXTRAS_DIR = Path(__file__).resolve().parents[1] / "extras"
sys.path.insert(0, str(EXTRAS_DIR))

from dokabun_extras import viz_embeddings  # noqa: E402


def test_collect_embedding_vectors_from_json(tmp_path: Path) -> None:
    df = pd.DataFrame({"eo": ["[0.1,0.2]", None]})

    dim, rows, vectors = viz_embeddings._collect_embedding_vectors(df, "eo", tmp_path)

    assert dim == 2
    assert rows == [0]
    assert vectors == [[0.1, 0.2]]


def test_collect_embedding_vectors_from_npy(tmp_path: Path) -> None:
    vector = np.array([0.5, 0.6], dtype="float32")
    file_path = tmp_path / "eof1_1.npy"
    np.save(file_path, vector)

    df = pd.DataFrame({"eo": [file_path.name]})

    dim, rows, vectors = viz_embeddings._collect_embedding_vectors(df, "eo", tmp_path)

    assert dim == 2
    assert rows == [0]
    assert vectors == [[0.5, 0.6]]


def test_collect_embedding_vectors_rejects_high_dim(tmp_path: Path) -> None:
    df = pd.DataFrame({"eo": ["[0,1,2,3]"]})

    with pytest.raises(ValueError, match="次元が大きすぎます"):
        viz_embeddings._collect_embedding_vectors(df, "eo", tmp_path)


def test_collect_embedding_vectors_rejects_mixed_dim(tmp_path: Path) -> None:
    df = pd.DataFrame({"eo": ["[0,1]", "[0,1,2]"]})

    with pytest.raises(ValueError, match="次元が行によって異なります"):
        viz_embeddings._collect_embedding_vectors(df, "eo", tmp_path)


def test_infer_hover_column_fallback_when_classify_fails() -> None:
    df = pd.DataFrame({"foo": [1], "i_text": ["hello"]})

    assert viz_embeddings._infer_hover_column(df) == "i_text"
