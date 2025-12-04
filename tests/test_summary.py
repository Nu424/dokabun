from dokabun.core.summary import ExecutionSummary


def test_execution_summary_tracks_counts_and_tokens():
    summary = ExecutionSummary()
    summary.start(total_rows=2)
    summary.record_success(0, {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12, "total_cost": 0.002})
    summary.record_failure(1, "TestError", "boom")
    summary.finish()

    data = summary.to_dict()
    assert data["success_rows"] == 1
    assert data["failed_rows"] == 1
    assert data["prompt_tokens"] == 5
    assert data["completion_tokens"] == 7
    assert data["error_counts"]["TestError"] == 1

