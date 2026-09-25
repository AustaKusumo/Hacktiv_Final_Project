import pandas as pd

from datalens.analysis import build_dataset_context, create_chart, dataframe_profile, local_answer


def sample_dataframe():
    return pd.DataFrame(
        {
            "region": ["Jakarta", "Bandung", "Jakarta", "Medan"],
            "sales": [100.0, 150.0, None, 200.0],
            "profit": [10.0, 18.0, 7.0, 25.0],
        }
    )


def test_dataframe_profile_reports_quality_metrics():
    df = sample_dataframe()

    profile = dataframe_profile(df)

    assert profile["rows"] == 4
    assert profile["columns"] == 3
    assert profile["missing_cells"] == 1
    assert profile["missing_by_column"] == {"sales": 1}
    assert profile["numeric_columns"] == ["sales", "profit"]


def test_dataset_context_limits_rows_and_is_serializable():
    context = build_dataset_context(sample_dataframe(), max_columns=2, max_sample_rows=2)

    assert '"rows": 4' in context
    assert '"name": "sales"' in context
    assert '"name": "profit"' not in context
    assert "Showing 2 of 3 columns and 2 of 4 rows" in context


def test_local_answer_handles_missing_values():
    answer = local_answer(sample_dataframe(), "Kolom mana yang punya missing value?")

    assert "sales" in answer
    assert "1" in answer


def test_local_answer_summarizes_named_numeric_column():
    answer = local_answer(sample_dataframe(), "Tolong ringkas profit")

    assert "profit" in answer
    assert "15.00" in answer


def test_create_chart_builds_correlation_heatmap():
    result = create_chart(sample_dataframe(), "Buat heatmap korelasi")

    assert result is not None
    assert result.figure.data[0].type == "heatmap"


def test_create_chart_returns_none_without_visual_intent():
    assert create_chart(sample_dataframe(), "Berapa jumlah baris?") is None

