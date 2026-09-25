"""Safe, deterministic dataset analysis used by the DataLens chatbot."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


MAX_CONTEXT_COLUMNS = 30
MAX_SAMPLE_ROWS = 5


@dataclass
class ChartResult:
    """A generated chart together with a human-readable description."""

    figure: go.Figure
    caption: str


def dataframe_profile(df: pd.DataFrame) -> dict[str, Any]:
    """Return a JSON-serializable profile without mutating the dataframe."""
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in df.columns if column not in numeric_columns]
    missing = df.isna().sum().sort_values(ascending=False)

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_by_column": {
            str(column): int(value)
            for column, value in missing.items()
            if int(value) > 0
        },
    }


def build_dataset_context(
    df: pd.DataFrame,
    max_columns: int = MAX_CONTEXT_COLUMNS,
    max_sample_rows: int = MAX_SAMPLE_ROWS,
) -> str:
    """Build a compact dataset summary suitable for an LLM prompt."""
    limited = df.iloc[:, :max_columns]
    profile = dataframe_profile(limited)
    column_info = [
        {
            "name": str(column),
            "dtype": str(limited[column].dtype),
            "unique": int(limited[column].nunique(dropna=True)),
            "missing": int(limited[column].isna().sum()),
        }
        for column in limited.columns
    ]

    numeric_summary: dict[str, dict[str, float | None]] = {}
    for column in limited.select_dtypes(include="number").columns:
        series = limited[column].dropna()
        numeric_summary[str(column)] = {
            "min": _safe_float(series.min()) if not series.empty else None,
            "mean": _safe_float(series.mean()) if not series.empty else None,
            "median": _safe_float(series.median()) if not series.empty else None,
            "max": _safe_float(series.max()) if not series.empty else None,
        }

    sample_rows = limited.head(max_sample_rows).where(pd.notna(limited), None)
    payload = {
        "profile": profile,
        "columns": column_info,
        "numeric_summary": numeric_summary,
        "sample_rows": sample_rows.to_dict(orient="records"),
        "context_note": (
            f"Showing {len(limited.columns)} of {len(df.columns)} columns and "
            f"{min(max_sample_rows, len(df))} of {len(df)} rows."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, default=str, indent=2)


def local_answer(df: pd.DataFrame, question: str) -> str:
    """Answer common questions deterministically when Gemini is unavailable."""
    normalized = question.casefold()
    profile = dataframe_profile(df)

    if any(token in normalized for token in ("missing", "kosong", "null", "na ")):
        if not profile["missing_by_column"]:
            return "Tidak ada missing value pada dataset ini."
        details = ", ".join(
            f"**{column}**: {count}"
            for column, count in profile["missing_by_column"].items()
        )
        return f"Missing value ditemukan pada {details}."

    if any(token in normalized for token in ("duplikat", "duplicate")):
        return f"Dataset memiliki **{profile['duplicate_rows']} baris duplikat**."

    if any(token in normalized for token in ("berapa baris", "jumlah baris", "shape", "ukuran data")):
        return (
            f"Dataset terdiri dari **{profile['rows']:,} baris** dan "
            f"**{profile['columns']} kolom**."
        )

    if any(token in normalized for token in ("kolom apa", "daftar kolom", "nama kolom")):
        return "Kolom yang tersedia: " + ", ".join(f"`{column}`" for column in df.columns)

    if any(token in normalized for token in ("korelasi", "correlation", "berhubungan")):
        numeric = df.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            return "Diperlukan setidaknya dua kolom numerik untuk menghitung korelasi."
        correlation = numeric.corr().abs()
        mask = correlation.where(~pd.DataFrame(
            [[row >= column for column in range(len(correlation))] for row in range(len(correlation))],
            index=correlation.index,
            columns=correlation.columns,
        ))
        pair = mask.stack().sort_values(ascending=False)
        if pair.empty:
            return "Korelasi tidak dapat dihitung dari nilai yang tersedia."
        (first, second), value = pair.index[0], pair.iloc[0]
        return (
            f"Hubungan linear terkuat terdapat antara **{first}** dan **{second}** "
            f"dengan |r| = **{value:.2f}**. Korelasi tidak selalu berarti sebab-akibat."
        )

    mentioned = _mentioned_columns(df, question)
    if mentioned and pd.api.types.is_numeric_dtype(df[mentioned[0]]):
        column = mentioned[0]
        series = df[column].dropna()
        if series.empty:
            return f"Kolom **{column}** tidak memiliki nilai numerik yang dapat dianalisis."
        return (
            f"Ringkasan **{column}**: rata-rata **{series.mean():.2f}**, "
            f"median **{series.median():.2f}**, minimum **{series.min():.2f}**, "
            f"dan maksimum **{series.max():.2f}**."
        )

    return (
        "Saya bisa membantu mengecek ukuran data, nama kolom, missing value, duplikat, "
        "korelasi, distribusi, dan ringkasan kolom. Tambahkan Gemini API key untuk "
        "pertanyaan analitis yang lebih bebas."
    )


def create_chart(df: pd.DataFrame, question: str) -> ChartResult | None:
    """Create a chart when the question contains a recognizable visual intent."""
    normalized = question.casefold()
    chart_tokens = (
        "grafik", "chart", "plot", "visual", "distribusi", "histogram",
        "korelasi", "correlation", "hubungan", "scatter", "bandingkan",
    )
    if not any(token in normalized for token in chart_tokens):
        return None

    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in df.columns if column not in numeric]
    mentioned = _mentioned_columns(df, question)
    mentioned_numeric = [column for column in mentioned if column in numeric]

    if any(token in normalized for token in ("korelasi", "correlation", "heatmap")):
        if len(numeric) < 2:
            return None
        corr = df[numeric[:12]].corr()
        figure = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                colorscale="Purples",
                zmin=-1,
                zmax=1,
                text=corr.round(2).values,
                texttemplate="%{text}",
            )
        )
        figure.update_layout(title="Correlation heatmap", height=480)
        return ChartResult(figure, "Heatmap korelasi untuk kolom numerik.")

    if any(token in normalized for token in ("hubungan", "scatter", "relationship")):
        columns = mentioned_numeric[:2] or numeric[:2]
        if len(columns) < 2:
            return None
        figure = px.scatter(
            df,
            x=columns[0],
            y=columns[1],
            opacity=0.72,
            title=f"Hubungan {columns[0]} dan {columns[1]}",
            color_discrete_sequence=["#7657FF"],
        )
        return ChartResult(figure, f"Scatter plot **{columns[0]}** terhadap **{columns[1]}**.")

    if any(token in normalized for token in ("bandingkan", "berdasarkan", "per ")):
        numeric_column = mentioned_numeric[0] if mentioned_numeric else (numeric[0] if numeric else None)
        mentioned_category = next((column for column in mentioned if column in categorical), None)
        category_column = mentioned_category or (categorical[0] if categorical else None)
        if numeric_column and category_column and df[category_column].nunique() <= 30:
            figure = px.box(
                df,
                x=category_column,
                y=numeric_column,
                color=category_column,
                title=f"{numeric_column} berdasarkan {category_column}",
            )
            figure.update_layout(showlegend=False)
            return ChartResult(
                figure,
                f"Perbandingan **{numeric_column}** berdasarkan **{category_column}**.",
            )

    column = mentioned_numeric[0] if mentioned_numeric else (numeric[0] if numeric else None)
    if column:
        figure = px.histogram(
            df,
            x=column,
            nbins=30,
            title=f"Distribusi {column}",
            color_discrete_sequence=["#7657FF"],
        )
        return ChartResult(figure, f"Distribusi nilai pada kolom **{column}**.")
    return None


def _mentioned_columns(df: pd.DataFrame, question: str) -> list[str]:
    normalized = question.casefold()
    matches: list[str] = []
    for column in df.columns:
        column_text = str(column)
        pattern = rf"(?<!\w){re.escape(column_text.casefold())}(?!\w)"
        if re.search(pattern, normalized):
            matches.append(column_text)
    return matches


def _safe_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None

