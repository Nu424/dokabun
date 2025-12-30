"""dokabun CLI エントリーポイント。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

from dokabun.config import AppConfig
from dokabun.core.runner import run as run_core
from dokabun.logging_utils import configure_logging, get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """dokabun CLI の引数パーサーを構築する。

    Returns:
        argparse.ArgumentParser: dokabun CLI 向けに設定済みのパーサー。
    """

    parser = argparse.ArgumentParser(
        prog="dokabun",
        description="スプレッドシートを入力として LLM で未入力セルを埋めるツール",
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="入力スプレッドシートのパス (.xlsx / .csv)",
    )
    parser.add_argument(
        "--model", default="openai/gpt-4.1-mini", help="使用するモデル名"
    )
    parser.add_argument("--temperature", type=float, default=None, help="モデル温度")
    parser.add_argument(
        "--max-tokens", dest="max_tokens", type=int, default=None, help="最大トークン数"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="同時並列実行数 (default: 5)",
    )
    parser.add_argument(
        "--max-rows",
        dest="max_rows",
        type=int,
        default=None,
        help="今回の実行で処理する最大行数",
    )
    parser.add_argument(
        "--partial-interval",
        dest="partial_interval",
        type=int,
        default=100,
        help="一時保存を行う行数間隔 (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="出力ファイルを保存するディレクトリ (default: 入力ファイルと同じ)",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default="INFO",
        help="ログレベル (INFO/DEBUG/WARNING/ERROR)",
    )
    parser.add_argument(
        "--timestamp",
        help="既存の実行を再開する際のタイムスタンプ。未指定時は最新を推定",
    )
    parser.add_argument(
        "--max-text-file-bytes",
        dest="max_text_file_bytes",
        type=int,
        default=262_144,
        help="テキストファイル読み込み時の最大サイズ（バイト） (default: 262144)",
    )
    parser.add_argument(
        "--nsf-ext",
        dest="nsf_ext",
        choices=["txt", "md"],
        default="txt",
        help="nsf_ 列で保存するファイルの拡張子 (default: txt)",
    )
    parser.add_argument(
        "--nsf-name-template",
        dest="nsf_name_template",
        default="nsf{nsf_index}_{row_no}.{ext}",
        help="nsf_ 出力ファイルの命名テンプレート",
    )
    parser.add_argument(
        "--nsf-name-template-filetarget",
        dest="nsf_name_template_filetarget",
        default="{target_file_stem}_nsf{nsf_index}.{ext}",
        help="t_ が単一ファイルパスの場合の命名テンプレート",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """dokabun CLI を実行するエントリーポイント。

    Args:
        argv: コマンドライン引数。``None`` の場合は ``sys.argv[1:]`` を利用。

    Returns:
        int: ``sys.exit`` に渡せる終了コード。
    """

    parser = build_parser()
    args = parser.parse_args(argv)

    load_dotenv()
    try:
        config = _build_config_from_args(args)
    except Exception as exc:  # noqa: BLE001
        parser.error(str(exc))
        return 2

    configure_logging(config.log_level, config.log_file)
    api_key = _load_api_key()
    if not api_key:
        print(
            "OPENROUTER_API_KEY が設定されていません。環境変数を確認してください。",
            file=sys.stderr,
        )
        return 1

    try:
        run_core(config, api_key)
    except KeyboardInterrupt:
        logger.warning("ユーザーにより処理が中断されました。")
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.exception("処理中に予期せぬエラーが発生しました。")
        print(f"エラー: {exc}", file=sys.stderr)
        return 1

    return 0


def _build_config_from_args(args: argparse.Namespace) -> AppConfig:
    """CLI 引数を AppConfig インスタンスへ変換する。

    Args:
        args: ``argparse`` でパース済みの名前空間。

    Returns:
        AppConfig: 正規化されたアプリ設定。
    """

    input_path = Path(args.input).expanduser()
    output_dir = (
        Path(args.output_dir).expanduser() if args.output_dir else input_path.parent
    )
    timestamp = args.timestamp or _detect_latest_timestamp(input_path, output_dir)

    data = {
        "input_path": input_path,
        "output_dir": output_dir,
        "timestamp": timestamp,
        "partial_interval": args.partial_interval,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_concurrency": args.concurrency,
        "max_rows": args.max_rows,
        "log_level": args.log_level,
        "max_text_file_bytes": args.max_text_file_bytes,
        "nsf_ext": args.nsf_ext,
        "nsf_name_template": args.nsf_name_template,
        "nsf_name_template_filetarget": args.nsf_name_template_filetarget,
    }
    return AppConfig.from_dict(data)


def _detect_latest_timestamp(input_path: Path, output_dir: Path) -> str | None:
    """対象スプレッドシートに対応する最新タイムスタンプを取得する。

    Args:
        input_path: 入力スプレッドシートのパス。
        output_dir: タイムスタンプ付き成果物を格納するディレクトリ。

    Returns:
        str | None: 見つかった場合はタイムスタンプ文字列。存在しなければ ``None``。
    """

    stem = input_path.stem
    pattern = f"{stem}_*.meta.json"
    meta_files = list(output_dir.glob(pattern))
    if not meta_files:
        return None

    latest = max(meta_files, key=lambda path: path.stat().st_mtime)
    return _extract_timestamp_from_meta(latest.name, stem)


def _extract_timestamp_from_meta(filename: str, stem: str) -> str | None:
    """メタファイル名からタイムスタンプ部分だけを抜き出す。

    Args:
        filename: メタファイル名（例: ``input_20250101_120000.meta.json``）。
        stem: 入力ファイルのベース名。

    Returns:
        str | None: 取得に成功した場合はタイムスタンプ。失敗時は ``None``。
    """

    prefix = f"{stem}_"
    suffix = ".meta.json"
    if not filename.startswith(prefix) or not filename.endswith(suffix):
        return None
    return filename[len(prefix) : -len(suffix)]


def _load_api_key(env: Iterable[tuple[str, str]] | None = None) -> str | None:
    """環境変数から OpenRouter API キーを取得する。

    Args:
        env: テスト用に環境変数を差し替える場合のキー・値 iterable。

    Returns:
        str | None: 見つかった場合は API キー。未設定なら ``None``。
    """

    data = dict(env) if env is not None else os.environ
    return data.get("OPENROUTER_API_KEY") or data.get("OPENAI_API_KEY")
