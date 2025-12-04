import base64
from pathlib import Path

from dokabun.preprocess import run_preprocess_pipeline
from dokabun.preprocess.image import ImagePreprocess
from dokabun.preprocess.text import PlainTextPreprocess


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

