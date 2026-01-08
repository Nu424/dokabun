from dokabun.config import AppConfig


def test_app_config_preserves_base_url_and_timestamp(tmp_path):
    cfg = AppConfig.from_dict(
        {
            "input_path": tmp_path / "dummy.xlsx",
            "output_dir": tmp_path,
            "base_url": "https://example.com/api/",
        }
    )

    assert cfg.base_url == "https://example.com/api"

    cfg2 = cfg.with_timestamp("20250101_000000")
    assert cfg2.base_url == cfg.base_url
    assert cfg2.timestamp == "20250101_000000"
