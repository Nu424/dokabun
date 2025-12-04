"""ロギング設定とロガー取得のユーティリティ。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

_configured = False
_log_file_path: Optional[Path] = None
_file_handler: Optional[logging.Handler] = None


def configure_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """ロギング設定を初期化する。

    Args:
        level: ログレベル文字列。
        log_file: ファイル出力を行う場合のパス。
    """

    global _configured, _log_file_path, _file_handler

    root_logger = logging.getLogger()
    level_value = level.upper()

    if _configured:
        root_logger.setLevel(level_value)
        _configure_file_handler(root_logger, log_file)
        return

    root_logger.setLevel(level_value)

    formatter = _build_formatter()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    _configure_file_handler(root_logger, log_file, formatter)

    _configured = True


def _configure_file_handler(
    root_logger: logging.Logger,
    log_file: Optional[Path],
    formatter: Optional[logging.Formatter] = None,
) -> None:
    """ファイルハンドラの付け外しを制御する。"""

    global _log_file_path, _file_handler

    if log_file is None:
        if _file_handler:
            root_logger.removeHandler(_file_handler)
            _file_handler.close()
            _file_handler = None
            _log_file_path = None
        return

    log_file = log_file.expanduser()
    if _file_handler and _log_file_path == log_file:
        return

    if _file_handler:
        root_logger.removeHandler(_file_handler)
        _file_handler.close()

    if formatter is None:
        formatter = _build_formatter()

    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    _file_handler = handler
    _log_file_path = log_file


def _build_formatter() -> logging.Formatter:
    """標準のログフォーマッタを返す。"""

    return logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    """指定した名前のロガーを返す。

    Args:
        name: ロガー名。

    Returns:
        logging.Logger: 設定済みロガー。
    """

    logger = logging.getLogger(name)
    if not _configured:
        configure_logging()
    # ルートロガーにのみハンドラを置く方針だが、ユーザーが直接 logger を
    # 生成してもハンドラが無限に増えないようにする。
    logger.propagate = True
    return logger
