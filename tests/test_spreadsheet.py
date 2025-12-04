import pandas as pd

from dokabun.config import AppConfig
from dokabun.io.spreadsheet import SpreadsheetReaderWriter


def test_spreadsheet_reader_writer_loads_and_resumes(tmp_path):
    input_path = tmp_path / "input.xlsx"
    df = pd.DataFrame({"t_content": ["hello"], "summary|本文の要約": [None]})
    df.to_excel(input_path, index=False)

    config = AppConfig.from_dict({"input_path": input_path, "output_dir": tmp_path})
    reader = SpreadsheetReaderWriter(config)
    loaded = reader.load()
    assert loaded.equals(df)

    loaded.loc[0, "summary|本文の要約"] = "done"
    reader.save_partial(loaded, 0, 0)
    reader.save_output(loaded)

    resume_config = AppConfig.from_dict(
        {"input_path": input_path, "output_dir": tmp_path, "timestamp": reader.timestamp}
    )
    resume_reader = SpreadsheetReaderWriter(resume_config)
    resumed = resume_reader.load()
    assert resumed.loc[0, "summary|本文の要約"] == "done"

    meta = resume_reader.load_meta_if_exists()
    assert meta and meta["last_completed_row"] == 0

