"""CLI for visualizing embedding vectors with Plotly."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from dokabun.core.runner.cells import is_empty_value
from dokabun.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_HOVER_MAX_CHARS = 200
DEFAULT_COLOR_SCALE = "Viridis"
DEFAULT_COLOR_SEQ = "Plotly"


def build_parser() -> argparse.ArgumentParser:
    """dokabun-viz CLI の引数パーサーを構築する。"""

    parser = argparse.ArgumentParser(
        prog="dokabun-viz",
        description="どかぶん処理後の埋め込み列を Plotly で可視化するツール",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="入力スプレッドシートのパス (.xlsx / .csv)",
    )
    parser.add_argument(
        "--embedding-col",
        required=True,
        help="埋め込みベクトル列名 (例: eo / eot2 / eou2 など)",
    )
    parser.add_argument(
        "--label-col",
        required=True,
        help="色分けに利用するラベル列名",
    )
    parser.add_argument(
        "--hover-col",
        default=None,
        help="ホバー表示に使う列名 (未指定時は i_ の先頭列)",
    )
    parser.add_argument(
        "--hover-max-chars",
        type=int,
        default=DEFAULT_HOVER_MAX_CHARS,
        help="ホバーテキストの最大文字数 (default: 200)",
    )
    parser.add_argument("--title", default=None, help="グラフタイトル (任意)")
    parser.add_argument(
        "--color-scale",
        default=DEFAULT_COLOR_SCALE,
        help="連続値ラベル用のカラースケール名 (default: Viridis)",
    )
    parser.add_argument(
        "--color-seq",
        default=DEFAULT_COLOR_SEQ,
        help="カテゴリラベル用のカラーパレット名、またはカンマ区切り色指定",
    )
    parser.add_argument(
        "--output-html",
        default=None,
        help="HTML 出力先パス (未指定時は入力と同じディレクトリに保存)",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="HTML 出力後にブラウザを開かない",
    )
    parser.add_argument(
        "--marker-size",
        type=int,
        default=15,
        help="マーカーのサイズ (default: 15)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """dokabun-viz CLI を実行するエントリーポイント。"""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except Exception as exc:  # noqa: BLE001
        print(f"エラー: {exc}", file=sys.stderr)
        return 1
    return 0


def run(args: argparse.Namespace) -> Path:
    """可視化処理を実行し、HTML を出力する。"""

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {input_path}")

    # ---入力ファイルをdfとして読み込む
    df = _load_dataframe(input_path)

    # ---各種列を解決する
    embedding_col = _resolve_column(df, args.embedding_col, "埋め込み列")
    label_col = _resolve_column(df, args.label_col, "ラベル列")

    hover_col = args.hover_col
    if hover_col:
        hover_col = _resolve_column(df, hover_col, "ホバー列")
    else:
        hover_col = _infer_hover_column(df)
        if hover_col is None:
            raise ValueError(
                "ホバー列が指定されておらず、i_ 列も見つかりません。"
                "--hover-col で指定してください。"
            )

    if args.hover_max_chars <= 0:
        raise ValueError("hover-max-chars は 1 以上で指定してください。")

    # ---埋め込みベクトルを収集する(次元取得・ファイル読み込みも)
    dim, row_indices, vectors = _collect_embedding_vectors(
        df, embedding_col, input_path.parent
    )

    # ---可視化用のDataFrameをつくる
    plot_df = _build_plot_dataframe(
        df,
        row_indices=row_indices,
        vectors=vectors,
        label_col=label_col,
        hover_col=hover_col,
        hover_max_chars=args.hover_max_chars,
        dim=dim,
    )

    title = args.title or f"Embedding Visualization ({embedding_col})"

    # ---PlotlyのFigureをつくる
    fig = _build_plotly_figure(
        plot_df,
        dim=dim,
        label_col=label_col,
        title=title,
        color_scale=args.color_scale,
        color_seq=args.color_seq,
        marker_size=args.marker_size,
    )

    # ---HTMLを出力する
    output_path = _resolve_output_path(input_path, args.output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    logger.info("HTML を出力しました: %s", output_path)

    if not args.no_open:
        webbrowser.open(output_path.resolve().as_uri())

    return output_path


def _load_dataframe(path: Path) -> pd.DataFrame:
    """入力ファイルを読み込み DataFrame を返す。"""

    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"未対応のファイル形式です: {path.suffix}")


def _resolve_column(df: pd.DataFrame, name: str, label: str) -> str:
    """列名を解決し、大文字小文字の揺れにも対応する。"""

    if name in df.columns:
        return name

    matches = [
        col
        for col in df.columns
        if isinstance(col, str) and col.lower() == name.lower()
    ]
    if len(matches) == 1:
        return matches[0]
    if matches:
        raise ValueError(f"{label} '{name}' に大小違いの候補が複数あります: {matches}")

    columns = ", ".join(str(col) for col in df.columns)
    raise ValueError(f"{label}が見つかりません: {name} (列一覧: {columns})")


def _infer_hover_column(df: pd.DataFrame) -> str | None:
    """hover 列を i_ から推定する。"""
    for column in df.columns:
        if isinstance(column, str) and column.lower().startswith("i_"):
            return column
    return None


def _collect_embedding_vectors(
    df: pd.DataFrame,
    embedding_col: str,
    base_dir: Path,
) -> tuple[int, list[int], list[list[float]]]:
    """埋め込み列を読み取り、次元を自動判定して返す。"""

    dim: int | None = None
    row_indices: list[int] = []
    vectors: list[list[float]] = []

    for row_index, value in df[embedding_col].items():
        if is_empty_value(value):
            continue
        try:
            vector = _parse_embedding_value(value, base_dir)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                f"埋め込み列 '{embedding_col}' の行 {row_index + 1} の値を解釈できません: {exc}"
            ) from exc

        if vector is None:
            continue

        if dim is None:
            dim = len(vector)
            _validate_dimension(dim)
        elif len(vector) != dim:
            raise ValueError(
                "埋め込み列の次元が行によって異なります。"
                f"行 {row_index + 1} は {len(vector)}次元、期待は {dim}次元です。"
            )

        row_indices.append(row_index)
        vectors.append(vector)

    if dim is None:
        raise ValueError(
            f"埋め込み列 '{embedding_col}' に有効なベクトルが見つかりません。"
        )

    return dim, row_indices, vectors


def _parse_embedding_value(value: object, base_dir: Path) -> list[float] | None:
    """埋め込みセル値をパースし、ベクトルとして返す。"""

    if is_empty_value(value):
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # ---.npyなら、ファイルを読み込んでベクトルとして返す
        if text.lower().endswith(".npy"):
            path = Path(text)
            if not path.is_absolute():
                path = (base_dir / path).resolve()
            if not path.exists():
                raise FileNotFoundError(f".npy ファイルが見つかりません: {path}")
            array = np.load(path)
            if array.ndim != 1:
                raise ValueError(".npy の埋め込みが 1 次元ではありません。")
            return _coerce_vector(array)

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("JSON 配列として解釈できません。") from exc
        return _coerce_vector(data)

    if isinstance(value, (list, tuple, np.ndarray)):
        return _coerce_vector(value)

    raise ValueError(f"未対応の埋め込み形式です: {type(value).__name__}")


def _coerce_vector(data: object) -> list[float]:
    """配列状のデータを float ベクトルに変換する。"""

    try:
        array = np.asarray(data, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("ベクトルの要素が数値ではありません。") from exc

    if array.ndim != 1:
        raise ValueError("ベクトルは 1 次元の配列である必要があります。")
    if array.size == 0:
        raise ValueError("ベクトルが空です。")
    return array.astype(float).tolist()


def _validate_dimension(dim: int) -> None:
    """次元が 2 または 3 かを検証する。"""

    if dim < 2:
        raise ValueError(f"次元が小さすぎます: {dim} (2 または 3 のみ対応)")
    if dim > 3:
        raise ValueError(f"次元が大きすぎます: {dim} (2 または 3 のみ対応)")


def _build_plot_dataframe(
    df: pd.DataFrame,
    *,
    row_indices: Iterable[int],
    vectors: list[list[float]],
    label_col: str,
    hover_col: str,
    hover_max_chars: int,
    dim: int,
) -> pd.DataFrame:
    """可視化用の DataFrame を構築する。"""

    hover_values = [
        _truncate_text(df.loc[idx, hover_col], hover_max_chars) for idx in row_indices
    ]
    label_values = [df.loc[idx, label_col] for idx in row_indices]
    coords = np.asarray(vectors, dtype=float)

    data = {
        label_col: label_values,
        "_hover_text": hover_values,
        "x": coords[:, 0],
        "y": coords[:, 1],
    }
    if dim == 3:
        data["z"] = coords[:, 2]

    return pd.DataFrame(data)


def _truncate_text(value: object, max_chars: int) -> str:
    """文字列を最大長でトリミングする。"""

    if is_empty_value(value):
        return ""
    text = str(value)
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def _resolve_output_path(input_path: Path, output_html: str | None) -> Path:
    """HTML の出力先パスを決定する。"""

    if output_html:
        return Path(output_html).expanduser()
    return input_path.parent / f"{input_path.stem}.dokabun_viz.html"


def _build_plotly_figure(
    plot_df: pd.DataFrame,
    *,
    dim: int,
    label_col: str,
    title: str,
    color_scale: str,
    color_seq: str,
    marker_size: int = 15,
):
    """Plotly の Figure を構築する。"""

    try:
        import plotly.express as px
    except ImportError as exc:
        raise RuntimeError(
            "plotly がインストールされていません。"
            "uv/pip で `dokabun[viz]` をインストールしてください。"
        ) from exc

    color_kwargs: dict[str, object] = {}
    if is_numeric_dtype(plot_df[label_col]):
        color_kwargs["color_continuous_scale"] = _resolve_color_scale(color_scale, px)
    else:
        color_kwargs["color_discrete_sequence"] = _resolve_color_sequence(color_seq, px)

    fig = None
    if dim == 2:
        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            color=label_col,
            hover_name="_hover_text",
            title=title,
            **color_kwargs,
        )
    else:
        fig = px.scatter_3d(
            plot_df,
            x="x",
            y="y",
            z="z",
            color=label_col,
            hover_name="_hover_text",
            title=title,
            **color_kwargs,
        )

    fig.update_traces(marker=dict(size=marker_size))
    return fig


def _resolve_color_scale(value: str, px: object) -> object:
    """連続カラースケールを解決する。"""

    text = value.strip()
    if "," in text:  # 色をカンマ区切りで複数指定した場合
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if parts:
            return parts
        return DEFAULT_COLOR_SCALE

    for palette in ("sequential", "diverging", "cyclical"):
        colors = getattr(px.colors, palette, None)
        if colors and hasattr(colors, text):
            return getattr(colors, text)
    return text


def _resolve_color_sequence(value: str, px: object) -> list[str]:
    """カテゴリ用カラーパレットを解決する。"""

    text = value.strip()
    if "," in text:
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if parts:
            return parts
        return list(px.colors.qualitative.Plotly)

    if hasattr(px.colors.qualitative, text):
        return list(getattr(px.colors.qualitative, text))

    return [text]
