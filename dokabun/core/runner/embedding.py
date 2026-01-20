"""Embedding helpers for generation and post-processing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from dokabun.logging_utils import get_logger
from dokabun.target import Target, TextTarget

logger = get_logger(__name__)

EMBEDDING_CELL_CHAR_LIMIT = 32767  # 埋め込みベクトルをセルに出力する場合の文字数制限
EMBEDDING_FILE_EXT = "npy"  # 埋め込みベクトルをファイルに出力する場合のファイル拡張子


def build_embedding_text(targets: Sequence[Target]) -> str:
    """埋め込み用の入力テキストを構築する。"""

    text_parts = [target.text for target in targets if isinstance(target, TextTarget)]
    if not text_parts:
        raise ValueError("埋め込み用のテキストが存在しません。")
    return "\n\n".join(text_parts)


def extract_embedding_vector(response: object) -> list[float]:
    """埋め込み API のレスポンスからベクトルを取り出す。"""

    data = getattr(response, "data", None)
    if not data:
        raise ValueError("埋め込み応答に data が含まれていません。")
    first = data[0]
    embedding = getattr(first, "embedding", None)
    if embedding is None and isinstance(first, dict):
        embedding = first.get("embedding")
    if embedding is None:
        raise ValueError("埋め込み応答に embedding が含まれていません。")
    return list(embedding)


def serialize_embedding_vector(vector: Sequence[float]) -> str:
    """埋め込みベクトルをセル書き込み用にシリアライズする。"""

    # テキストを小さくするため、カンマとコロンのみを使用(不要なスペースを削除)
    return json.dumps(list(vector), ensure_ascii=False, separators=(",", ":"))


def write_embedding_file(path: Path, vector: Sequence[float]) -> None:
    """埋め込みベクトルをファイルへ保存する。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(vector, dtype="float32")
    np.save(path, array)


def build_eof_filename(*, embedding_index: int, row_no: int) -> str:
    """埋め込みベクトル出力ファイルの名前を生成する。"""

    return f"eof{embedding_index}_{row_no}.{EMBEDDING_FILE_EXT}"


def prepare_embedding_output(
    vector: Sequence[float],
    *,
    output_dir: Path,
    row_no: int,
    embedding_index: int,
    force_file: bool,
) -> str:
    """埋め込みベクトルをセルまたはファイルへ出力する。"""

    serialized = serialize_embedding_vector(vector)
    if not force_file and len(serialized) <= EMBEDDING_CELL_CHAR_LIMIT:
        return serialized

    filename = build_eof_filename(embedding_index=embedding_index, row_no=row_no)
    output_path = (output_dir / filename).resolve()
    write_embedding_file(output_path, vector)
    if not force_file:
        logger.info(
            "埋め込み出力が長すぎるためファイル出力にフォールバックしました: %s",
            filename,
        )
    return filename


def apply_embedding_reductions(
    *,
    df: object,
    embedding_vectors: dict[str, dict[int, list[float]]],
    embedding_spec_map: dict[str, dict[str, object]],
    embedding_index_map: dict[str, int],
    output_dir: Path,
) -> None:
    """後段の次元削減を実行して DataFrame を更新する。"""

    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df は pandas.DataFrame である必要があります。")

    for column, row_vectors in embedding_vectors.items():
        spec = embedding_spec_map.get(column, {})
        post_method = spec.get("post_method")
        post_dim = spec.get("post_dim")
        if not post_method or not post_dim:
            continue

        row_indices = sorted(row_vectors.keys())
        vectors = [row_vectors[row_index] for row_index in row_indices]
        try:
            # ---後処理(次元削減)を実行する
            reduced_vectors = reduce_embeddings(
                vectors, method=str(post_method), dim=int(post_dim)
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("埋め込みの後段次元削減に失敗しました: %s (%s)", column, exc)
            continue

        # ---後処理後の埋め込みベクトルを、セルまたはファイルに出力する
        for offset, row_index in enumerate(row_indices):
            output_value = prepare_embedding_output(
                reduced_vectors[offset],
                output_dir=output_dir,
                row_no=row_index + 1,
                embedding_index=embedding_index_map.get(column, 1),
                force_file=bool(spec.get("file_output")),
            )
            df.loc[row_index, column] = output_value


def reduce_embeddings(
    vectors: Sequence[Sequence[float]],
    *,
    method: str,
    dim: int,
) -> list[list[float]]:
    """埋め込みベクトルを指定の手法で次元削減する。"""

    # ---前確認
    if dim <= 0:
        raise ValueError("後段次元数が不正です。")
    matrix = np.asarray(vectors, dtype="float32")
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("埋め込みベクトルが空です。")
    n_samples, n_features = matrix.shape
    if dim > n_features:
        raise ValueError("後段次元数が元の次元数を超えています。")

    method = method.lower()
    if method == "p":  # PCA
        if n_samples < 2:
            logger.warning("PCA を実行できないため先頭切り出しで代替します。")
            return truncate_embeddings(matrix, dim)
        from sklearn.decomposition import PCA

        reducer = PCA(n_components=dim, random_state=0)
        reduced = reducer.fit_transform(matrix)
        return reduced.astype("float32").tolist()
    if method == "t":  # t-SNE
        if n_samples < 2:
            logger.warning("t-SNE を実行できないため先頭切り出しで代替します。")
            return truncate_embeddings(matrix, dim)
        from sklearn.manifold import TSNE

        perplexity = min(30, max(1, n_samples - 1))
        reducer = TSNE(
            n_components=dim,
            perplexity=perplexity,
            random_state=0,
            init="pca",
        )
        reduced = reducer.fit_transform(matrix)
        return reduced.astype("float32").tolist()
    if method == "u":  # UMAP
        if n_samples < 3:
            logger.warning("UMAP を実行できないため先頭切り出しで代替します。")
            return truncate_embeddings(matrix, dim)
        try:
            import umap
        except ImportError:
            logger.warning(
                "UMAP を利用するには `umap-learn` の追加インストールが必要です。"
                "（例: `uv sync --extra umap` / `uvx --from . \"dokabun[umap]\" ...`）"
                "先頭切り出しで代替します。"
            )
            return truncate_embeddings(matrix, dim)

        n_neighbors = min(15, n_samples - 1)
        reducer = umap.UMAP(
            n_components=dim,
            n_neighbors=n_neighbors,
            random_state=0,
        )
        reduced = reducer.fit_transform(matrix)
        return reduced.astype("float32").tolist()

    raise ValueError(f"未対応の後段次元削減方式です: {method}")


def truncate_embeddings(matrix: np.ndarray, dim: int) -> list[list[float]]:
    """埋め込みを先頭から切り出して指定次元に合わせる。"""

    if dim > matrix.shape[1]:
        raise ValueError("後段次元数が元の次元数を超えています。")
    return matrix[:, :dim].astype("float32").tolist()
