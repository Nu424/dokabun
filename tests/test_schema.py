from dokabun.llm.schema import build_schema_from_headers


def test_build_schema_from_headers_generates_properties():
    headers = ["summary|本文の要約", "sentiment"]
    schema_payload = build_schema_from_headers(headers, name="row")

    json_schema = schema_payload["schema"]
    assert json_schema["type"] == "object"
    assert json_schema["properties"]["summary"]["description"] == "本文の要約"
    assert "sentiment" in json_schema["required"]

