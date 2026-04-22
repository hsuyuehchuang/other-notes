#!/usr/bin/env python3
"""
Asset performance report generator (TW/US stocks & ETFs).

Core behavior:
- Parse symbols from the first column of section: "### ETF 實例說明（常見標的）"
  in loan-stock.md (default), unless --symbols is provided.
- Compute N-year summary CAGR per symbol.
- Plot normalized cumulative return (%), where each symbol starts at 0%.
- Generate markdown block + interactive chart HTML (+ optional PNG) + HTML report.
- Upsert managed auto block in markdown (no overwrite of manual table).
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import plotly.graph_objects as go
import yfinance as yf


DEFAULT_SYMBOLS = [
    ("0050", "0050.TW"),
    ("0050 正二", "00631L.TW"),
    ("006208", "006208.TW"),
    ("SPY", "SPY"),
    ("QQQ", "QQQ"),
    ("VOO", "VOO"),
    ("VTI", "VTI"),
    ("VT", "VT"),
]

SECTION_HEADER = "### ETF 實例說明（常見標的）"
GENERATED_SECTION_HEADER = "### ETF 年化報酬快照（自動更新）"
BEGIN_MARKER = "<!-- ETF_CAGR_AUTO:BEGIN -->"
END_MARKER = "<!-- ETF_CAGR_AUTO:END -->"
DEFAULT_MD_PATH = Path(__file__).with_name("loan-stock.md")

# Professional finance-style palette (muted, high contrast)
PLOT_COLORS = [
    "#1F4E79",  # deep blue
    "#2E7D32",  # green
    "#C62828",  # red
    "#EF6C00",  # orange
    "#6A1B9A",  # purple
    "#00897B",  # teal
    "#546E7A",  # blue gray
    "#B8860B",  # dark gold
    "#AD1457",  # magenta
    "#0277BD",  # cyan blue
]


@dataclass
class Row:
    name: str
    input_symbol: str
    ticker: str
    total_return_pct: str
    total_return_rank: str
    annualized_pct: str
    period: str
    method: str
    split_events: str


@dataclass
class AssetData:
    close: pd.Series | None
    splits: pd.Series | None


DATA_CACHE: Dict[str, AssetData] = {}


def _scalar(v) -> float:
    return float(v.iloc[0]) if hasattr(v, "iloc") else float(v)


def _is_ascii(s: str) -> bool:
    try:
        s.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def make_plot_label(name: str, input_symbol: str, ticker: str) -> str:
    clean_name = name.strip()
    clean_input = input_symbol.strip().replace(" ", "")
    ticker_base = ticker.split(".")[0].strip()
    if clean_name and _is_ascii(clean_name):
        return clean_name
    if clean_input and _is_ascii(clean_input):
        return clean_input
    return ticker_base if ticker_base else ticker


def years_ago(today: dt.date, years: int) -> dt.date:
    try:
        return today.replace(year=today.year - years)
    except ValueError:
        return today.replace(month=2, day=28, year=today.year - years)


def cagr(start_price: float, end_price: float, years: float) -> float:
    if start_price <= 0 or end_price <= 0 or years <= 0:
        return float("nan")
    return (end_price / start_price) ** (1.0 / years) - 1.0


def _extract_column_series(df: pd.DataFrame, col_name: str) -> pd.Series | None:
    if df is None or df.empty:
        return None
    if col_name in df.columns:
        s = df[col_name]
    elif isinstance(df.columns, pd.MultiIndex):
        selected = [c for c in df.columns if c[-1] == col_name]
        if not selected:
            return None
        s = df[selected[0]]
    else:
        return None
    if isinstance(s, pd.DataFrame):
        if s.empty:
            return None
        s = s.iloc[:, 0]
    s = s.dropna()
    return s if not s.empty else None


def _normalize_splits_series(splits: pd.Series | None) -> pd.Series | None:
    if splits is None:
        return None
    s = splits.copy()
    if isinstance(s, pd.DataFrame):
        if s.empty:
            return None
        s = s.iloc[:, 0]
    s = s.dropna()
    if s.empty:
        return None
    s = s[s != 0]
    if s.empty:
        return None
    idx = pd.to_datetime(s.index, utc=True, errors="coerce")
    valid = ~idx.isna()
    if not valid.any():
        return None
    s = s.loc[valid]
    idx = idx[valid].tz_convert(None).normalize()
    s.index = idx
    s = s.sort_index()
    # Multiple split entries on same date should multiply.
    s = s.groupby(s.index).prod()
    return s


def _merge_split_series(parts: List[pd.Series | None]) -> pd.Series | None:
    normalized = [p for p in parts if p is not None and not p.empty]
    if not normalized:
        return None
    raw = pd.concat(normalized).sort_index()
    merged: Dict[pd.Timestamp, float] = {}
    for idx, grp in raw.groupby(level=0):
        unique_vals: List[float] = []
        for v in grp.values:
            fv = float(v)
            if fv <= 0:
                continue
            if not any(abs(fv - u) <= 1e-8 for u in unique_vals):
                unique_vals.append(fv)
        if not unique_vals:
            continue
        factor = 1.0
        for u in unique_vals:
            factor *= u
        merged[pd.Timestamp(idx)] = factor
    if not merged:
        return None
    return pd.Series(merged).sort_index()


def _build_total_return_close(
    close_raw: pd.Series,
    dividends_raw: pd.Series | None,
    splits: pd.Series | None,
) -> pd.Series:
    close = close_raw.dropna().copy()
    if close.empty:
        return close
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()

    div = (
        dividends_raw.reindex(close.index).fillna(0.0)
        if dividends_raw is not None
        else pd.Series(0.0, index=close.index)
    )
    split_map = _normalize_splits_series(splits)

    future_factor: Dict[pd.Timestamp, float] = {}
    cum = 1.0
    for d in reversed(close.index):
        future_factor[d] = cum
        if split_map is not None and d in split_map.index:
            cum *= float(split_map.loc[d])
    factor = pd.Series(future_factor).sort_index()

    adj_close = close / factor
    adj_div = div / factor
    base = adj_close.shift(1)
    div_yield = (adj_div / base).fillna(0.0)
    ret = adj_close.pct_change().fillna(0.0) + div_yield
    ret.iloc[0] = 0.0
    return (1.0 + ret).cumprod() * float(adj_close.iloc[0])


def _infer_missing_splits(close_raw: pd.Series, known_splits: pd.Series | None) -> pd.Series | None:
    close = close_raw.dropna().copy()
    if close.empty:
        return None
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    ret = close / close.shift(1) - 1.0

    known_idx = set() if known_splits is None else set(pd.to_datetime(known_splits.index))
    inferred: Dict[pd.Timestamp, float] = {}
    for d, r in ret.items():
        if pd.isna(r) or r > -0.60:
            continue
        if d in known_idx:
            continue
        prev = float(close.loc[:d].iloc[-2])
        cur = float(close.loc[d])
        if cur <= 0:
            continue
        ratio = prev / cur
        if ratio >= 1.5:
            inferred[pd.Timestamp(d)] = ratio

    if not inferred:
        return None
    s = pd.Series(inferred).sort_index()
    return _normalize_splits_series(s)


def _download_asset_data(ticker: str) -> AssetData:
    df = yf.download(ticker, period="max", auto_adjust=False, actions=True, progress=False)
    if df is None or df.empty:
        return AssetData(close=None, splits=None)
    close_raw = _extract_column_series(df, "Close")
    if close_raw is None:
        return AssetData(close=None, splits=None)
    adj_close_raw = _extract_column_series(df, "Adj Close")
    base_close = adj_close_raw if adj_close_raw is not None else close_raw
    splits_from_download = _normalize_splits_series(_extract_column_series(df, "Stock Splits"))
    splits_from_ticker = None
    try:
        splits_from_ticker = _normalize_splits_series(yf.Ticker(ticker).splits)
    except Exception:
        splits_from_ticker = None

    known_splits = _merge_split_series([splits_from_download, splits_from_ticker])
    inferred_splits = _infer_missing_splits(base_close, known_splits)
    merged_splits = _merge_split_series([known_splits, inferred_splits])
    # Use Adj Close as baseline (already split/dividend adjusted when available),
    # and only patch missing corporate actions through inferred splits.
    total_return_close = _build_total_return_close(base_close, dividends_raw=None, splits=inferred_splits)
    return AssetData(close=total_return_close, splits=merged_splits)


def get_asset_data(ticker: str) -> AssetData:
    cached = DATA_CACHE.get(ticker)
    if cached is not None:
        return cached
    data = _download_asset_data(ticker)
    DATA_CACHE[ticker] = data
    return data


def _download_close_series(ticker: str):
    return get_asset_data(ticker).close


def build_alias_map() -> dict[str, str]:
    alias: dict[str, str] = {}
    for name, symbol in DEFAULT_SYMBOLS:
        alias[name.strip().upper()] = symbol
        alias[name.replace(" ", "").strip().upper()] = symbol
    alias["0050正二"] = "00631L"
    alias["BTC"] = "BTC-USD"
    alias["XBT"] = "BTC-USD"
    return alias


def resolve_ticker(raw_symbol: str) -> str | None:
    s = raw_symbol.strip()
    if not s:
        return None

    alias = build_alias_map()
    key = s.strip().upper()
    key_no_space = s.replace(" ", "").strip().upper()
    if key in alias:
        s = alias[key]
    elif key_no_space in alias:
        s = alias[key_no_space]

    if "." in s:
        return s

    starts_with_digit = s[:1].isdigit()
    is_numeric_like = s.isdigit()
    if is_numeric_like or starts_with_digit:
        candidates = [f"{s}.TW", f"{s}.TWO", s]
    else:
        up = s.upper()
        candidates = [up, f"{up}.TW", f"{up}.TWO"]

    for c in candidates:
        close = _download_close_series(c)
        if close is not None:
            return c
    return None


def format_split_ratio(v: float) -> str:
    if v <= 0:
        return "N/A"
    if v >= 1:
        return f"{v:g}x"
    reverse = 1.0 / v
    return f"1/{reverse:g}x"


def summarize_splits(splits: pd.Series | None, start_date: dt.date, end_date: dt.date) -> str:
    if splits is None or splits.empty:
        return "無"
    window = splits[(splits.index.date >= start_date) & (splits.index.date <= end_date)]
    if window.empty:
        return "無"
    items = [f"{idx.date()} ({format_split_ratio(float(val))})" for idx, val in window.items()]
    return "; ".join(items)


def parse_symbols_from_section_table(md_path: Path) -> List[str]:
    text = md_path.read_text(encoding="utf-8")
    anchor = text.find(SECTION_HEADER)
    if anchor == -1:
        return []

    chunk = text[anchor:].splitlines()
    symbols: List[str] = []
    in_table = False
    for line in chunk[1:]:
        raw = line.strip()
        if not raw:
            if in_table:
                break
            continue
        if not raw.startswith("|"):
            if in_table:
                break
            continue

        in_table = True
        cols = [c.strip() for c in raw.strip("|").split("|")]
        if not cols:
            continue
        first = cols[0]
        if first in {"代號", ":---", "---", ""}:
            continue
        if first.startswith(":") and first.endswith(":"):
            continue
        symbols.append(first)

    seen = set()
    ordered: List[str] = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def parse_symbols_arg(symbols_arg: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for token in symbols_arg.split(","):
        t = token.strip()
        if not t:
            continue
        if ":" in t:
            name, sym = t.split(":", 1)
            out.append((name.strip(), sym.strip()))
        elif "=" in t:
            name, sym = t.split("=", 1)
            out.append((name.strip(), sym.strip()))
        else:
            out.append((t, t))
    return out


def build_symbol_targets(symbols_arg: str | None) -> List[Tuple[str, str, str]]:
    base: Sequence[Tuple[str, str]]
    if symbols_arg and symbols_arg.strip():
        base = parse_symbols_arg(symbols_arg)
    else:
        base = DEFAULT_SYMBOLS
    targets: List[Tuple[str, str, str]] = []
    for name, symbol in base:
        resolved = resolve_ticker(symbol) or symbol
        targets.append((name, symbol, resolved))
    return targets


def build_symbol_targets_from_md(md_path: Path) -> List[Tuple[str, str, str]]:
    symbols = parse_symbols_from_section_table(md_path)
    targets: List[Tuple[str, str, str]] = []
    for s in symbols:
        resolved = resolve_ticker(s) or s
        targets.append((s, s, resolved))
    return targets


def calc_summary_row(name: str, input_symbol: str, ticker: str, start_target: dt.date, lookback_years: int) -> Row:
    data = get_asset_data(ticker)
    close = data.close
    if close is None:
        return Row(name, input_symbol, ticker, "N/A", "N/A", "N/A", "N/A", "無資料", "N/A")

    first = close.index[0].date()
    last = close.index[-1].date()
    end_price = _scalar(close.iloc[-1])

    if first <= start_target:
        s = close[close.index.date >= start_target]
        if s.empty:
            start_date = first
            start_price = _scalar(close.iloc[0])
            method = "可得資料以來"
        else:
            start_date = s.index[0].date()
            start_price = _scalar(s.iloc[0])
            method = f"近{lookback_years}年"
    else:
        start_date = first
        start_price = _scalar(close.iloc[0])
        method = "可得資料以來"

    years = (last - start_date).days / 365.2425
    val = cagr(start_price, end_price, years)
    pct = f"{val * 100:.2f}%" if not math.isnan(val) else "N/A"
    total_return = (end_price / start_price - 1.0) if start_price > 0 else float("nan")
    total_return_pct = f"{total_return * 100:.2f}%" if not math.isnan(total_return) else "N/A"
    period = f"{start_date} ～ {last}"
    split_events = summarize_splits(data.splits, start_date, last)
    return Row(name, input_symbol, ticker, total_return_pct, "N/A", pct, period, method, split_events)


def normalize_to_zero(close, start_target: dt.date) -> pd.Series | None:
    if close is None or close.empty:
        return None
    s = close[close.index.date >= start_target]
    if s.empty:
        return None
    base = _scalar(s.iloc[0])
    if base <= 0:
        return None
    return (s / base - 1.0) * 100.0


def yearly_returns(close, start_target: dt.date) -> Dict[int, float]:
    if close is None or close.empty:
        return {}
    s = close[close.index.date >= start_target]
    if s.empty:
        return {}
    out: Dict[int, float] = {}
    for y, grp in s.groupby(s.index.year):
        if grp.empty:
            continue
        start = _scalar(grp.iloc[0])
        end = _scalar(grp.iloc[-1])
        if start > 0:
            out[int(y)] = (end / start - 1.0) * 100.0
    return out


def _parse_pct_string(pct_text: str) -> float | None:
    s = (pct_text or "").strip()
    if not s or s.upper() == "N/A":
        return None
    try:
        return float(s.replace("%", ""))
    except ValueError:
        return None


def assign_total_return_rank(rows: List[Row]) -> None:
    scored: List[Tuple[int, float]] = []
    for i, r in enumerate(rows):
        v = _parse_pct_string(r.total_return_pct)
        if v is not None:
            scored.append((i, v))
    scored.sort(key=lambda x: x[1], reverse=True)
    total = len(scored)
    for rank, (idx, _) in enumerate(scored, start=1):
        rows[idx].total_return_rank = f"{rank}/{total}"


def write_plot_png(fig, stem: Path) -> Path | None:
    png_path = stem.with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)

    if importlib.util.find_spec("kaleido") is None:
        print(f"Skip {png_path.name}: install kaleido to export PNG.")
        return None

    fig.write_image(str(png_path))
    return png_path


def generate_normalized_return_chart(
    normalized_by_label: Dict[str, pd.Series],
    split_events_by_label: Dict[str, pd.Series],
    years: int,
) -> go.Figure | None:
    valid = {k: v for k, v in normalized_by_label.items() if v is not None and not v.empty}
    if not valid:
        return None

    fig = go.Figure()

    for i, (label, series) in enumerate(valid.items()):
        color = PLOT_COLORS[i % len(PLOT_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=label,
                line={"color": color, "width": 2},
                hovertemplate=f"{label}<br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}%<extra></extra>",
            )
        )
        splits = split_events_by_label.get(label)
        if splits is None or splits.empty:
            continue
        for d, _ in splits.items():
            if d < series.index[0] or d > series.index[-1]:
                continue
            fig.add_vline(x=d, line_color=color, line_dash="dot", opacity=0.35)

    fig.add_hline(y=0, line_color="#9E9E9E", line_dash="dash", opacity=0.85)
    fig.update_layout(
        template="plotly_white",
        title=f"Normalized Cumulative Return (Base = 0%, Lookback {years}Y)",
        xaxis_title="Year",
        yaxis_title="Return (%)",
        hovermode="x unified",
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.02,
            "font": {"size": 11},
        },
        margin={"l": 60, "r": 260, "t": 70, "b": 50},
        width=1280,
        height=720,
    )
    fig.update_xaxes(dtick="M12", tickformat="%Y", showgrid=True, gridcolor="#E5E7EB")
    fig.update_yaxes(ticksuffix="%", showgrid=True, gridcolor="#E5E7EB")
    return fig


def build_markdown_block(
    rows: Iterable[Row],
    years: int,
    chart_rel_path: str = "",
    html_rel_path: str = "",
) -> str:
    lines: List[str] = []
    lines.append("| 名稱 | 輸入代號 | Yahoo 代號(解析後) | 總報酬（估算） | 總報酬排名 | 年化報酬（估算） | 計算區間 | 計算方式 | 期間內分割事件 |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in rows:
        lines.append(
            f"| {r.name} | {r.input_symbol} | {r.ticker} | {r.total_return_pct} | {r.total_return_rank} | {r.annualized_pct} | {r.period} | {r.method} | {r.split_events} |"
        )
    lines.append("")
    lines.append(
        f"註：使用 Yahoo Finance `Adj Close` 作為總報酬基準，並用 `Ticker.splits` 補齊缺漏分割；優先取近 {years} 年；"
        "若歷史不足則改用可得資料以來。"
    )
    lines.append("代號解析：若輸入純數字（如 `0050`、`2330`），會優先嘗試 `.TW`、`.TWO`。")
    lines.append("分割說明：分割事件已列出；`auto-adjust` 價格已做分割還原，長期走勢不會只由分割本身造成。")
    lines.append("公式：`(期末/期初)^(1/年數)-1`。")
    lines.append("")
    lines.append(f"主圖為「起點標準化累積報酬」（每檔基準日 = 0%），比較基準一致。")
    if chart_rel_path:
        lines.append("")
        lines.append(f"![近{years}年標準化累積報酬圖]({chart_rel_path})")
    if html_rel_path:
        lines.append("")
        lines.append(f"[開啟完整 HTML 報表（含每年報酬表）]({html_rel_path})")
    return "\n".join(lines)


def build_html_report(
    rows: List[Row],
    yearly_returns_by_label: Dict[str, Dict[int, float]],
    years: int,
    chart_embed_html: str,
    chart_png_rel_path_from_html: str,
) -> str:
    # Summary table
    summary_df = pd.DataFrame(
        [
            {
                "Name": r.name,
                "Input": r.input_symbol,
                "Ticker": r.ticker,
                "Total Return": r.total_return_pct,
                "Total Return Rank": r.total_return_rank,
                "Annualized": r.annualized_pct,
                "Period": r.period,
                "Method": r.method,
                "Split Events (In Period)": r.split_events,
            }
            for r in rows
        ]
    )

    # Yearly return matrix
    all_years = sorted({y for d in yearly_returns_by_label.values() for y in d.keys()})
    yr_df = pd.DataFrame(index=all_years)
    for label, yd in yearly_returns_by_label.items():
        yr_df[label] = [yd.get(y, math.nan) for y in all_years]
    yr_df.index.name = "Year"
    yr_df = yr_df.apply(lambda col: col.map(lambda x: f"{x:.2f}%" if pd.notna(x) else ""))

    summary_html = summary_df.to_html(index=False, escape=False, table_id="summary-table")
    yearly_html = yr_df.to_html(index=True, escape=False, table_id="yearly-table")

    gen_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Asset Performance Report ({years}Y)</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 24px; color: #222; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .meta {{ color: #666; margin-bottom: 18px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 14px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
    th {{ background: #f5f7fa; }}
    th.sortable {{ cursor: pointer; user-select: none; position: relative; padding-right: 20px; }}
    th.sortable::after {{ content: "↕"; position: absolute; right: 6px; color: #9aa0a6; font-size: 12px; }}
    th.sortable[data-order="asc"]::after {{ content: "↑"; color: #222; }}
    th.sortable[data-order="desc"]::after {{ content: "↓"; color: #222; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
    .note {{ color: #555; font-size: 13px; }}
    .table-wrap {{ overflow-x: auto; }}
  </style>
</head>
<body>
  <h1>Asset Performance Report ({years}Y)</h1>
  <div class="meta">Generated at: {gen_time}</div>

  <h2>1) Normalized Cumulative Return (Base = 0%)</h2>
  <p class="note">All symbols start from the same 0% baseline at the lookback start date.</p>
  {chart_embed_html}
  {'<p class="note">Static snapshot:</p><img src="' + chart_png_rel_path_from_html + '" alt="normalized-cumulative-return" />' if chart_png_rel_path_from_html else ''}

  <h2>2) CAGR Summary</h2>
  <div class="table-wrap">
  {summary_html}
  </div>

  <h2>3) Calendar-Year Return Table</h2>
  <p class="note">Each cell is the return within that calendar year.</p>
  <div class="table-wrap">
  {yearly_html}
  </div>

  <script>
    function parseCellValue(text) {{
      const t = (text || "").trim();
      if (!t) return null;
      if (/^-?\\d+(\\.\\d+)?%$/.test(t)) return parseFloat(t.replace("%", ""));
      if (/^-?\\d+(\\.\\d+)?$/.test(t.replace(/,/g, ""))) return parseFloat(t.replace(/,/g, ""));
      if (/^\\d+\\/\\d+$/.test(t)) {{
        const p = t.split("/").map(Number);
        return p[1] !== 0 ? p[0] / p[1] : p[0];
      }}
      const d = Date.parse(t);
      if (!Number.isNaN(d)) return d;
      return t.toLowerCase();
    }}

    function sortTableByColumn(table, colIndex, order) {{
      const tbody = table.tBodies[0];
      if (!tbody) return;
      const rows = Array.from(tbody.rows);
      rows.sort((a, b) => {{
        const va = parseCellValue((a.cells[colIndex] || {{ innerText: "" }}).innerText);
        const vb = parseCellValue((b.cells[colIndex] || {{ innerText: "" }}).innerText);
        if (va === null && vb === null) return 0;
        if (va === null) return 1;
        if (vb === null) return -1;
        if (typeof va === "number" && typeof vb === "number") {{
          return order === "asc" ? va - vb : vb - va;
        }}
        const sa = String(va);
        const sb = String(vb);
        return order === "asc" ? sa.localeCompare(sb) : sb.localeCompare(sa);
      }});
      rows.forEach((row) => tbody.appendChild(row));
    }}

    function makeSortable(tableId) {{
      const table = document.getElementById(tableId);
      if (!table || !table.tHead || !table.tHead.rows.length) return;
      const headers = Array.from(table.tHead.rows[0].cells);
      headers.forEach((th, idx) => {{
        th.classList.add("sortable");
        th.dataset.order = "none";
        th.addEventListener("click", () => {{
          headers.forEach((h) => {{ if (h !== th) h.dataset.order = "none"; }});
          const nextOrder = th.dataset.order === "asc" ? "desc" : "asc";
          th.dataset.order = nextOrder;
          sortTableByColumn(table, idx, nextOrder);
        }});
      }});
    }}

    makeSortable("summary-table");
    makeSortable("yearly-table");
  </script>
</body>
</html>
"""
    return html


def _build_generated_block(content: str) -> str:
    return f"{BEGIN_MARKER}\n{content.rstrip()}\n{END_MARKER}\n"


def upsert_generated_section(md_path: Path, generated_content: str) -> None:
    text = md_path.read_text(encoding="utf-8")
    block = _build_generated_block(generated_content)

    begin = text.find(BEGIN_MARKER)
    end = text.find(END_MARKER)
    if begin != -1 and end != -1 and end > begin:
        end_line = text.find("\n", end)
        end_idx = len(text) if end_line == -1 else end_line + 1
        updated = text[:begin] + block + text[end_idx:]
        md_path.write_text(updated, encoding="utf-8")
        return

    anchor = text.find(SECTION_HEADER)
    if anchor == -1:
        sep = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
        section = f"{GENERATED_SECTION_HEADER}\n\n{block}"
        md_path.write_text(text + sep + section, encoding="utf-8")
        return

    next_h3 = text.find("\n### ", anchor + len(SECTION_HEADER))
    insert_at = len(text) if next_h3 == -1 else next_h3
    prefix = text[:insert_at]
    suffix = text[insert_at:]
    if not prefix.endswith("\n\n"):
        prefix += "\n" if prefix.endswith("\n") else "\n\n"
    section = f"{GENERATED_SECTION_HEADER}\n\n{block}"
    updated = prefix + section + ("\n" if suffix and not suffix.startswith("\n") else "") + suffix
    md_path.write_text(updated, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Asset performance report (TW/US stocks & ETFs)")
    p.add_argument("--years", type=int, default=20, help="lookback years (default: 20)")
    p.add_argument(
        "--symbols",
        type=str,
        default="",
        help='comma-separated symbols, e.g. "0050,SPY,AAPL,2330" or "台積電:2330,Apple:AAPL"',
    )
    p.add_argument(
        "--symbols-from-md",
        type=str,
        default="",
        help="markdown file path to parse symbols from ETF example table first column",
    )
    p.add_argument(
        "--write-md",
        type=str,
        default="",
        help="optional markdown path to upsert managed report section",
    )
    p.add_argument("--no-plot", action="store_true", help="disable chart generation")
    p.add_argument("--no-html", action="store_true", help="disable html report generation")
    p.add_argument(
        "--plot-output",
        type=str,
        default="",
        help="optional png output path; default ../figure/asset-normalized-{N}y.png",
    )
    p.add_argument(
        "--html-output",
        type=str,
        default="",
        help="optional html output path; default ../figure/asset-report-{N}y.html",
    )
    args = p.parse_args()

    start_target = years_ago(dt.date.today(), args.years)

    if args.symbols.strip():
        targets = build_symbol_targets(args.symbols)
    else:
        md_source = (
            Path(args.symbols_from_md).resolve()
            if args.symbols_from_md.strip()
            else (Path(args.write_md).resolve() if args.write_md.strip() else DEFAULT_MD_PATH.resolve())
        )
        if md_source.exists():
            targets = build_symbol_targets_from_md(md_source)
            if not targets:
                targets = build_symbol_targets("")
        else:
            targets = build_symbol_targets("")

    rows: List[Row] = []
    normalized_by_label: Dict[str, pd.Series] = {}
    split_events_by_label: Dict[str, pd.Series] = {}
    yearly_by_label: Dict[str, Dict[int, float]] = {}
    used_labels: set[str] = set()

    for name, raw, ticker in targets:
        row = calc_summary_row(name, raw, ticker, start_target, args.years)
        rows.append(row)
        data = get_asset_data(ticker)
        close = data.close
        if close is None or close.empty:
            continue

        label = make_plot_label(name, raw, ticker)
        base = label
        k = 2
        while label in used_labels:
            label = f"{base}_{k}"
            k += 1
        used_labels.add(label)

        norm = normalize_to_zero(close, start_target)
        if norm is not None and not norm.empty:
            normalized_by_label[label] = norm
        if data.splits is not None and not data.splits.empty:
            s = data.splits[data.splits.index.date >= start_target]
            if not s.empty:
                split_events_by_label[label] = s
        yearly_by_label[label] = yearly_returns(close, start_target)

    assign_total_return_rank(rows)

    chart_rel_path = ""
    html_rel_path = ""

    if args.write_md:
        md_path = Path(args.write_md).resolve()
        if args.plot_output.strip():
            plot_output_path = Path(args.plot_output).resolve()
        else:
            plot_output_path = (md_path.parent.parent / "figure" / f"asset-normalized-{args.years}y.png").resolve()
        plot_stem = plot_output_path.with_suffix("") if plot_output_path.suffix.lower() == ".png" else plot_output_path

        if args.html_output.strip():
            html_path = Path(args.html_output).resolve()
        else:
            html_path = (md_path.parent.parent / "figure" / f"asset-report-{args.years}y.html").resolve()

        chart_embed_html = ""
        chart_png_path: Path | None = None
        if not args.no_plot:
            fig = generate_normalized_return_chart(
                normalized_by_label,
                split_events_by_label,
                args.years,
            )
            if fig is not None:
                chart_embed_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
                chart_png_path = write_plot_png(fig, plot_stem)
                if chart_png_path is not None:
                    chart_rel_path = os.path.relpath(chart_png_path, md_path.parent).replace("\\", "/")

        if not args.no_html:
            html_path.parent.mkdir(parents=True, exist_ok=True)
            chart_png_rel_for_html = (
                os.path.relpath(chart_png_path, html_path.parent).replace("\\", "/")
                if chart_png_path is not None
                else ""
            )
            html_doc = build_html_report(
                rows,
                yearly_by_label,
                args.years,
                chart_embed_html,
                chart_png_rel_for_html,
            )
            html_path.write_text(html_doc, encoding="utf-8")
            html_rel_path = os.path.relpath(html_path, md_path.parent).replace("\\", "/")

    md_block = build_markdown_block(
        rows,
        args.years,
        chart_rel_path=chart_rel_path,
        html_rel_path=html_rel_path,
    )
    print(md_block)

    if args.write_md:
        md_path = Path(args.write_md).resolve()
        upsert_generated_section(md_path, md_block)
        print(f"\n[OK] Updated: {md_path}")


if __name__ == "__main__":
    main()
