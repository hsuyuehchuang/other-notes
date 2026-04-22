#!/usr/bin/env python3
"""
Asset performance report generator (TW/US stocks & ETFs).

Core behavior:
- Parse symbols from the first column of section: "### ETF 實例說明（常見標的）"
  in loan-stock.md (default), unless --symbols is provided.
- Compute N-year summary CAGR per symbol.
- Plot normalized cumulative return (%), where each symbol starts at 0%.
- Generate markdown block + PNG chart + HTML report.
- Upsert managed auto block in markdown (no overwrite of manual table).
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
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


def configure_matplotlib_style() -> None:
    # Keep labels in English to avoid CJK glyph issues on environments without CJK fonts.
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["axes.facecolor"] = "#FAFAFA"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.edgecolor"] = "#9E9E9E"
    plt.rcParams["grid.color"] = "#D0D0D0"
    plt.rcParams["grid.alpha"] = 0.35


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


def _download_asset_data(ticker: str) -> AssetData:
    df = yf.download(ticker, period="max", auto_adjust=True, actions=True, progress=False)
    if df is None or df.empty:
        return AssetData(close=None, splits=None)
    close = _extract_column_series(df, "Close")
    splits = _extract_column_series(df, "Stock Splits")
    if splits is not None:
        splits = splits[splits != 0]
        if splits.empty:
            splits = None
    return AssetData(close=close, splits=splits)


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
        return Row(name, input_symbol, ticker, "N/A", "N/A", "無資料", "N/A")

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
    period = f"{start_date} ～ {last}"
    split_events = summarize_splits(data.splits, start_date, last)
    return Row(name, input_symbol, ticker, pct, period, method, split_events)


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


def generate_normalized_return_chart(
    normalized_by_label: Dict[str, pd.Series],
    split_events_by_label: Dict[str, pd.Series],
    years: int,
    output_path: Path,
) -> bool:
    valid = {k: v for k, v in normalized_by_label.items() if v is not None and not v.empty}
    if not valid:
        return False
    configure_matplotlib_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6), dpi=160)
    ax = plt.gca()

    for i, (label, series) in enumerate(valid.items()):
        color = PLOT_COLORS[i % len(PLOT_COLORS)]
        ax.plot(series.index, series.values, linewidth=2.0, color=color, alpha=0.95, label=label)
        splits = split_events_by_label.get(label)
        if splits is None or splits.empty:
            continue
        for d, _ in splits.items():
            if d < series.index[0] or d > series.index[-1]:
                continue
            ax.axvline(d, color=color, linestyle=":", linewidth=1.0, alpha=0.35)

    ax.axhline(0, color="#9E9E9E", linestyle="--", linewidth=1.0, alpha=0.85)
    ax.set_title(f"Normalized Cumulative Return (Base = 0%, Lookback {years}Y)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Return (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="best", frameon=False, ncol=2, fontsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    return True


def build_markdown_block(
    rows: Iterable[Row],
    years: int,
    chart_rel_path: str = "",
    html_rel_path: str = "",
) -> str:
    lines: List[str] = []
    lines.append("| 名稱 | 輸入代號 | Yahoo 代號(解析後) | 年化報酬（估算） | 計算區間 | 計算方式 | 期間內分割事件 |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in rows:
        lines.append(
            f"| {r.name} | {r.input_symbol} | {r.ticker} | {r.annualized_pct} | {r.period} | {r.method} | {r.split_events} |"
        )
    lines.append("")
    lines.append(
        f"註：使用 Yahoo Finance 調整後收盤價（auto-adjust）估算 CAGR，優先取近 {years} 年；"
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
    chart_rel_path_from_html: str,
) -> str:
    # Summary table
    summary_df = pd.DataFrame(
        [
            {
                "Name": r.name,
                "Input": r.input_symbol,
                "Ticker": r.ticker,
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
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
    .note {{ color: #555; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>Asset Performance Report ({years}Y)</h1>
  <div class="meta">Generated at: {gen_time}</div>

  <h2>1) Normalized Cumulative Return (Base = 0%)</h2>
  <p class="note">All symbols start from the same 0% baseline at the lookback start date.</p>
  <img src="{chart_rel_path_from_html}" alt="normalized-cumulative-return" />

  <h2>2) CAGR Summary</h2>
  {summary_df.to_html(index=False, escape=False)}

  <h2>3) Calendar-Year Return Table</h2>
  <p class="note">Each cell is the return within that calendar year.</p>
  {yr_df.to_html(index=True, escape=False)}
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

    chart_rel_path = ""
    html_rel_path = ""

    if args.write_md:
        md_path = Path(args.write_md).resolve()
        if args.plot_output.strip():
            plot_path = Path(args.plot_output).resolve()
        else:
            plot_path = (md_path.parent.parent / "figure" / f"asset-normalized-{args.years}y.png").resolve()

        if args.html_output.strip():
            html_path = Path(args.html_output).resolve()
        else:
            html_path = (md_path.parent.parent / "figure" / f"asset-report-{args.years}y.html").resolve()

        if not args.no_plot:
            ok_plot = generate_normalized_return_chart(
                normalized_by_label,
                split_events_by_label,
                args.years,
                plot_path,
            )
            if ok_plot:
                chart_rel_path = os.path.relpath(plot_path, md_path.parent).replace("\\", "/")

        if not args.no_html:
            html_path.parent.mkdir(parents=True, exist_ok=True)
            chart_rel_for_html = os.path.relpath(plot_path, html_path.parent).replace("\\", "/")
            html_doc = build_html_report(rows, yearly_by_label, args.years, chart_rel_for_html)
            html_path.write_text(html_doc, encoding="utf-8")
            html_rel_path = os.path.relpath(html_path, md_path.parent).replace("\\", "/")

    md_block = build_markdown_block(rows, args.years, chart_rel_path=chart_rel_path, html_rel_path=html_rel_path)
    print(md_block)

    if args.write_md:
        md_path = Path(args.write_md).resolve()
        upsert_generated_section(md_path, md_block)
        print(f"\n[OK] Updated: {md_path}")


if __name__ == "__main__":
    main()
