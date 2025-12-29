import base64
from pathlib import Path

import pytest

from dokabun.preprocess import build_default_preprocessors, run_preprocess_pipeline
from dokabun.preprocess.image import ImagePreprocess
from dokabun.preprocess.text import PlainTextPreprocess, TextFilePreprocess


def test_plain_text_preprocess_normalizes_whitespace(tmp_path):
    preprocess = PlainTextPreprocess()
    result = preprocess.preprocess("  こんにちは \r\n世界\r", tmp_path)
    assert result.text == "こんにちは \n世界"


def test_image_preprocess_reads_file(tmp_path):
    image_path = tmp_path / "sample.png"
    payload = b"fake-image-bytes"
    image_path.write_bytes(payload)

    preprocess = ImagePreprocess()
    target = preprocess.preprocess(str(image_path), tmp_path)

    assert target.mime_type == "image/png"
    assert target.base64_data == base64.b64encode(payload).decode("ascii")


def test_run_preprocess_pipeline_prefers_image(tmp_path):
    image_path = tmp_path / "example.jpg"
    image_path.write_bytes(b"dummy")

    target = run_preprocess_pipeline(str(image_path), tmp_path)
    assert target.to_llm_content()["type"] == "image_url"

    text_target = run_preprocess_pipeline("just text", tmp_path)
    assert text_target.to_llm_content()["type"] == "text"


def test_text_file_preprocess_reads_markdown_file(tmp_path):
    md_path = tmp_path / "note.md"
    content = "# タイトル\n\nこれはテストです。"
    md_path.write_text(content, encoding="utf-8")

    preprocess = TextFilePreprocess()
    target = preprocess.preprocess(str(md_path), tmp_path)

    assert target.to_llm_content()["type"] == "text"
    assert "# タイトル\n\nこれはテストです。" in target.text


def test_text_file_preprocess_reads_txt_file(tmp_path):
    txt_path = tmp_path / "document.txt"
    content = "Hello, World!\nThis is a test."
    txt_path.write_text(content, encoding="utf-8")

    preprocess = TextFilePreprocess()
    target = preprocess.preprocess(str(txt_path), tmp_path)

    assert target.to_llm_content()["type"] == "text"
    assert "Hello, World!" in target.text
    assert "This is a test." in target.text


def test_text_file_preprocess_handles_relative_path(tmp_path):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    txt_path = subdir / "file.txt"
    content = "Relative path test"
    txt_path.write_text(content, encoding="utf-8")

    preprocess = TextFilePreprocess()
    target = preprocess.preprocess("subdir/file.txt", tmp_path)

    assert target.text == "Relative path test"


def test_text_file_preprocess_raises_on_nonexistent_file(tmp_path):
    preprocess = TextFilePreprocess()
    with pytest.raises(FileNotFoundError):
        preprocess.preprocess("nonexistent.txt", tmp_path)


def test_text_file_preprocess_raises_on_file_too_large(tmp_path):
    txt_path = tmp_path / "large.txt"
    # 100バイトのファイルを作成
    content = "x" * 100
    txt_path.write_text(content, encoding="utf-8")

    preprocess = TextFilePreprocess(max_bytes=50)  # 50バイト制限
    with pytest.raises(ValueError, match="大きすぎます"):
        preprocess.preprocess(str(txt_path), tmp_path)


def test_text_file_preprocess_handles_utf8_encoding(tmp_path):
    txt_path = tmp_path / "utf8.txt"
    content = "日本語テスト\nUTF-8 encoding"
    txt_path.write_text(content, encoding="utf-8")

    preprocess = TextFilePreprocess()
    target = preprocess.preprocess(str(txt_path), tmp_path)

    assert "日本語テスト" in target.text
    assert "UTF-8 encoding" in target.text


def test_text_file_preprocess_handles_utf8_bom(tmp_path):
    txt_path = tmp_path / "utf8bom.txt"
    content = "BOM付きUTF-8"
    txt_path.write_text(content, encoding="utf-8-sig")

    preprocess = TextFilePreprocess()
    target = preprocess.preprocess(str(txt_path), tmp_path)

    assert "BOM付きUTF-8" in target.text


def test_run_preprocess_pipeline_prefers_text_file_over_plain_text(tmp_path):
    md_path = tmp_path / "readme.md"
    content = "# README\n\nThis is a markdown file."
    md_path.write_text(content, encoding="utf-8")

    target = run_preprocess_pipeline(str(md_path), tmp_path)
    assert target.to_llm_content()["type"] == "text"
    assert "# README" in target.text

    # ファイルパスではない通常のテキストは PlainTextPreprocess が処理
    text_target = run_preprocess_pipeline("just text", tmp_path)
    assert text_target.to_llm_content()["type"] == "text"
    assert text_target.text == "just text"


def test_build_default_preprocessors_includes_text_file_preprocess():
    preprocessors = build_default_preprocessors(max_text_file_bytes=1000)
    assert len(preprocessors) == 3
    assert isinstance(preprocessors[0], ImagePreprocess)
    assert isinstance(preprocessors[1], TextFilePreprocess)
    assert isinstance(preprocessors[2], PlainTextPreprocess)
    assert preprocessors[1].max_bytes == 1000

