#!/usr/bin/env python3
"""
Backward-compatible wrapper.

This script has been renamed to:
  asset_performance_report.py
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).with_name("asset_performance_report.py")
    runpy.run_path(str(target), run_name="__main__")
